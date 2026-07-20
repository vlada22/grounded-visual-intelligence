from pathlib import Path

import pytest
from pydantic import ValidationError

from gvi.adapters.sam3 import Sam3Adapter, Sam3FrameRecord, load_recorded_output
from gvi.scene_memory import SceneMemory

FIXTURE = Path(__file__).parent / "fixtures" / "sam3_tabletop_output.json"


def test_converts_recorded_sam3_output_to_tracks() -> None:
    recorded = load_recorded_output(FIXTURE)
    scene = Sam3Adapter(min_confidence=0.5).convert(recorded)

    assert scene.metadata.duration_s == pytest.approx(2.5)
    assert len(scene.tracks) == 1
    assert scene.tracks[0].track_id == "sam3:7"
    assert scene.tracks[0].concept == "red toy car"
    assert [observation.visible for observation in scene.tracks[0].observations] == [
        True,
        True,
        False,
        True,
        True,
    ]


def test_converts_normalized_xywh_boxes_to_pixel_xyxy() -> None:
    scene = Sam3Adapter().convert(load_recorded_output(FIXTURE))
    box = scene.tracks[0].observations[0].bbox

    assert box.x_min == pytest.approx(32.0)
    assert box.y_min == pytest.approx(144.0)
    assert box.x_max == pytest.approx(108.8)
    assert box.y_max == pytest.approx(208.8)


def test_preserves_model_and_prompt_provenance() -> None:
    scene = Sam3Adapter().convert(load_recorded_output(FIXTURE))

    assert scene.provenance[0].model_id == "facebook/sam3"
    assert scene.provenance[0].model_version == "3.1"
    assert scene.provenance[0].prompt == "red toy car"


def test_scene_queries_ignore_observations_below_visibility_threshold() -> None:
    memory = SceneMemory(Sam3Adapter().convert(load_recorded_output(FIXTURE)))

    assert memory.count_unique("toy car") == 1
    assert memory.first_seen("sam3:7").frame_indices == (0,)
    assert memory.last_seen("sam3:7").frame_indices == (4,)


def test_rejects_misaligned_sam3_output_columns() -> None:
    with pytest.raises(ValidationError, match="equal lengths"):
        Sam3FrameRecord.model_validate(
            {
                "frame_index": 0,
                "out_obj_ids": [1, 2],
                "out_probs": [0.9],
                "out_boxes_xywh": [
                    {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                    {"x": 0.5, "y": 0.5, "width": 0.2, "height": 0.2},
                ],
            }
        )


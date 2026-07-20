from pathlib import Path

import pytest
from pydantic import ValidationError

from gvi.adapters.grounded_sam2 import (
    GroundedSam2Adapter,
    GroundedSam2FrameRecord,
    load_grounded_sam2_output,
)

FIXTURE = Path(__file__).parent / "fixtures" / "grounded_sam2_tabletop_output.json"


def test_converts_grounded_sam2_records_to_scene_tracks() -> None:
    scene = GroundedSam2Adapter().convert(load_grounded_sam2_output(FIXTURE))

    assert scene.metadata.duration_s == pytest.approx(2.0)
    assert len(scene.tracks) == 1
    assert scene.tracks[0].track_id == "grounded-sam2:1"
    assert scene.tracks[0].concept == "red toy car"
    assert [observation.frame_index for observation in scene.tracks[0].observations] == [
        0,
        1,
        3,
    ]
    assert scene.tracks[0].observations[0].confidence == pytest.approx(0.91)


def test_records_detector_and_tracker_provenance_separately() -> None:
    scene = GroundedSam2Adapter().convert(load_grounded_sam2_output(FIXTURE))

    assert [entry.model_id for entry in scene.provenance] == [
        "IDEA-Research/GroundingDINO",
        "facebook/sam2",
    ]
    assert scene.provenance[0].prompt == "red toy car"
    assert scene.provenance[1].prompt is None


def test_rejects_non_positive_recorded_mask_area() -> None:
    with pytest.raises(ValidationError, match="positive area"):
        GroundedSam2FrameRecord(
            frame_index=0,
            object_ids=(1,),
            boxes_xyxy=({"x_min": 1, "y_min": 1, "x_max": 2, "y_max": 2},),
            mask_areas=(0,),
            mask_uris=("mask.rle.json",),
        )

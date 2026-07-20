from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gvi.inference.mask_rle import BinaryMaskRle, decode_binary_mask
from gvi.inference.sam3_worker import Sam3InferenceWorker, VideoDescriptor


class FakePredictor:
    def __init__(self, *, fail_during_propagation: bool = False) -> None:
        self.requests: list[dict[str, Any]] = []
        self.fail_during_propagation = fail_during_propagation

    def handle_request(self, request: dict[str, Any]) -> Mapping[str, Any]:
        self.requests.append(request)
        if request["type"] == "start_session":
            return {"session_id": "session-1"}
        return {}

    def handle_stream_request(
        self,
        request: dict[str, Any],
    ) -> Iterable[Mapping[str, Any]]:
        self.requests.append(request)
        if self.fail_during_propagation:
            raise RuntimeError("simulated propagation failure")
        return (
            {
                "frame_index": 0,
                "outputs": {
                    "out_obj_ids": np.asarray([7]),
                    "out_probs": np.asarray([0.95]),
                    "out_boxes_xywh": np.asarray([[0.1, 0.2, 0.3, 0.4]]),
                    "out_binary_masks": np.asarray([[[False, True], [True, False]]]),
                },
            },
            {
                "frame_index": 1,
                "outputs": {
                    "out_obj_ids": np.asarray([7]),
                    "out_probs": np.asarray([0.9]),
                    "out_boxes_xywh": np.asarray([[0.2, 0.2, 0.3, 0.4]]),
                    "out_binary_masks": np.asarray([[[True, True], [False, False]]]),
                },
            },
        )


def descriptor() -> VideoDescriptor:
    return VideoDescriptor(
        scene_id="worker-fixture",
        source_uri="prepared/worker-fixture.mp4",
        width=640,
        height=360,
        fps=2.0,
        frame_count=2,
    )


def test_worker_records_stream_and_materializes_masks(tmp_path: Path) -> None:
    predictor = FakePredictor()
    recorded = Sam3InferenceWorker(predictor).run(
        video_path=Path("prepared/worker-fixture.mp4"),
        video=descriptor(),
        prompt="red toy car",
        output_dir=tmp_path,
    )

    assert [request["type"] for request in predictor.requests] == [
        "start_session",
        "add_prompt",
        "propagate_in_video",
        "close_session",
    ]
    assert [frame.frame_index for frame in recorded.frames] == [0, 1]
    assert (tmp_path / "sam3-output.json").is_file()
    assert recorded.frames[0].out_mask_uris == ("masks/frame-00000-object-7.rle.json",)

    encoded = BinaryMaskRle.model_validate_json(
        (tmp_path / recorded.frames[0].out_mask_uris[0]).read_text(encoding="utf-8")
    )
    assert np.array_equal(
        decode_binary_mask(encoded),
        np.asarray([[False, True], [True, False]]),
    )


def test_worker_output_is_json_portable(tmp_path: Path) -> None:
    Sam3InferenceWorker(FakePredictor()).run(
        video_path=Path("prepared/worker-fixture.mp4"),
        video=descriptor(),
        prompt="red toy car",
        output_dir=tmp_path,
    )

    payload = json.loads((tmp_path / "sam3-output.json").read_text(encoding="utf-8"))
    assert payload["model_version"] == "3.1"
    assert payload["frames"][1]["out_obj_ids"] == [7]


def test_worker_closes_session_after_propagation_failure(tmp_path: Path) -> None:
    predictor = FakePredictor(fail_during_propagation=True)

    with pytest.raises(RuntimeError, match="simulated propagation failure"):
        Sam3InferenceWorker(predictor).run(
            video_path=Path("prepared/worker-fixture.mp4"),
            video=descriptor(),
            prompt="red toy car",
            output_dir=tmp_path,
        )

    assert predictor.requests[-1]["type"] == "close_session"


def test_worker_validates_prompt_frame_before_starting_session(tmp_path: Path) -> None:
    predictor = FakePredictor()

    with pytest.raises(ValueError, match="frame index"):
        Sam3InferenceWorker(predictor).run(
            video_path=Path("prepared/worker-fixture.mp4"),
            video=descriptor(),
            prompt="red toy car",
            prompt_frame_index=descriptor().frame_count,
            output_dir=tmp_path,
        )

    assert predictor.requests == []

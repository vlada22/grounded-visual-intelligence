from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from pydantic import Field

from gvi.adapters.sam3 import (
    NormalizedBoxXYWH,
    Sam3FrameRecord,
    Sam3RecordedOutput,
)
from gvi.inference.mask_rle import encode_binary_mask
from gvi.models import StrictModel


class Sam3Predictor(Protocol):
    def handle_request(self, request: dict[str, Any]) -> Mapping[str, Any]: ...

    def handle_stream_request(
        self,
        request: dict[str, Any],
    ) -> Iterable[Mapping[str, Any]]: ...


class VideoDescriptor(StrictModel):
    scene_id: str = Field(min_length=1)
    source_uri: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0.0)
    frame_count: int = Field(gt=0)


class Sam3InferenceWorker:
    """Owns the stateful SAM 3.1 video lifecycle and artifact materialization."""

    def __init__(
        self,
        predictor: Sam3Predictor,
        *,
        model_id: str = "facebook/sam3",
        model_version: str = "3.1",
    ) -> None:
        self.predictor = predictor
        self.model_id = model_id
        self.model_version = model_version

    def run(
        self,
        *,
        video_path: Path,
        video: VideoDescriptor,
        prompt: str,
        output_dir: Path,
        prompt_frame_index: int = 0,
    ) -> Sam3RecordedOutput:
        if not prompt.strip():
            raise ValueError("prompt must not be blank")
        if not 0 <= prompt_frame_index < video.frame_count:
            raise ValueError("prompt frame index must be within the described video frame range")

        output_dir.mkdir(parents=True, exist_ok=True)
        session = self.predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(video_path),
            }
        )
        session_id = str(session["session_id"])

        try:
            self.predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": prompt_frame_index,
                    "text": prompt,
                }
            )
            frames = tuple(
                self._record_frame(response, output_dir=output_dir)
                for response in self.predictor.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": session_id,
                    }
                )
            )
        finally:
            self.predictor.handle_request(
                {
                    "type": "close_session",
                    "session_id": session_id,
                }
            )

        recorded = Sam3RecordedOutput(
            scene_id=video.scene_id,
            source_uri=video.source_uri,
            prompt=prompt,
            model_id=self.model_id,
            model_version=self.model_version,
            width=video.width,
            height=video.height,
            fps=video.fps,
            frame_count=video.frame_count,
            frames=frames,
        )
        (output_dir / "sam3-output.json").write_text(
            recorded.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return recorded

    def _record_frame(
        self,
        response: Mapping[str, Any],
        *,
        output_dir: Path,
    ) -> Sam3FrameRecord:
        frame_index = int(response["frame_index"])
        outputs = response["outputs"]
        if not isinstance(outputs, Mapping):
            raise TypeError("SAM 3 response outputs must be a mapping")

        object_ids = tuple(int(value) for value in _to_python(outputs["out_obj_ids"]))
        probabilities = tuple(float(value) for value in _to_python(outputs["out_probs"]))
        raw_boxes = _to_python(outputs["out_boxes_xywh"])
        boxes = tuple(
            NormalizedBoxXYWH(x=box[0], y=box[1], width=box[2], height=box[3]) for box in raw_boxes
        )
        masks = _normalize_masks(outputs["out_binary_masks"], expected=len(object_ids))
        mask_uris = tuple(
            self._write_mask(
                mask,
                output_dir=output_dir,
                frame_index=frame_index,
                object_id=object_id,
            )
            for object_id, mask in zip(object_ids, masks, strict=True)
        )
        return Sam3FrameRecord(
            frame_index=frame_index,
            out_obj_ids=object_ids,
            out_probs=probabilities,
            out_boxes_xywh=boxes,
            out_mask_uris=mask_uris,
        )

    @staticmethod
    def _write_mask(
        mask: np.ndarray,
        *,
        output_dir: Path,
        frame_index: int,
        object_id: int,
    ) -> str:
        relative_path = Path("masks") / f"frame-{frame_index:05d}-object-{object_id}.rle.json"
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            encode_binary_mask(mask).model_dump_json(indent=2),
            encoding="utf-8",
        )
        return relative_path.as_posix()


def _to_python(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _normalize_masks(value: Any, *, expected: int) -> tuple[np.ndarray, ...]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()

    masks = np.asarray(value, dtype=np.bool_)
    if masks.ndim == 2 and expected == 1:
        masks = masks[np.newaxis, ...]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0, ...]
    if masks.ndim != 3 or masks.shape[0] != expected:
        raise ValueError(
            f"expected {expected} binary masks shaped [N,H,W] or [N,1,H,W]; received {masks.shape}"
        )
    return tuple(masks[index] for index in range(expected))


def build_official_predictor() -> Sam3Predictor:
    """Build SAM lazily so importing the evidence package never requires it."""

    try:
        from sam3.model_builder import build_sam3_multiplex_video_predictor
    except ImportError as error:
        raise RuntimeError(
            "SAM 3.1 is not installed. Follow the upstream gated-checkpoint installation guide."
        ) from error
    return build_sam3_multiplex_video_predictor()

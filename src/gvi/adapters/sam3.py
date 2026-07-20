from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import Field, model_validator

from gvi.models import (
    BoundingBox,
    InferenceProvenance,
    Observation,
    Scene,
    SceneMetadata,
    StrictModel,
    Track,
)


class NormalizedBoxXYWH(StrictModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_extent(self) -> NormalizedBoxXYWH:
        if self.x + self.width > 1.0 or self.y + self.height > 1.0:
            raise ValueError("normalized box must remain within the image")
        return self

    def to_pixel_box(self, *, image_width: int, image_height: int) -> BoundingBox:
        return BoundingBox(
            x_min=round(self.x * image_width, 6),
            y_min=round(self.y * image_height, 6),
            x_max=round((self.x + self.width) * image_width, 6),
            y_max=round((self.y + self.height) * image_height, 6),
        )


class Sam3FrameRecord(StrictModel):
    """Serializable form of one SAM 3.1 propagated frame response.

    The first three collections mirror SAM 3.1's official output names. Binary
    masks are materialized separately and referenced through ``out_mask_uris``
    because tensors do not belong in the evidence JSON document.
    """

    frame_index: int = Field(ge=0)
    out_obj_ids: tuple[int, ...]
    out_probs: tuple[float, ...]
    out_boxes_xywh: tuple[NormalizedBoxXYWH, ...]
    out_mask_uris: tuple[str | None, ...] = ()

    @model_validator(mode="after")
    def validate_columns(self) -> Sam3FrameRecord:
        expected = len(self.out_obj_ids)
        columns = (self.out_probs, self.out_boxes_xywh)
        if any(len(column) != expected for column in columns):
            raise ValueError("SAM 3 frame output columns must have equal lengths")
        if self.out_mask_uris and len(self.out_mask_uris) != expected:
            raise ValueError("out_mask_uris must be empty or align with object IDs")
        if len(self.out_obj_ids) != len(set(self.out_obj_ids)):
            raise ValueError("object IDs must be unique within a frame")
        if any(probability < 0.0 or probability > 1.0 for probability in self.out_probs):
            raise ValueError("output probabilities must be in the inclusive [0, 1] range")
        return self

    def mask_uri_at(self, index: int) -> str | None:
        return self.out_mask_uris[index] if self.out_mask_uris else None


class Sam3RecordedOutput(StrictModel):
    schema_version: str = "1.0"
    scene_id: str = Field(min_length=1)
    source_uri: str
    prompt: str = Field(min_length=1)
    model_id: str = "facebook/sam3"
    model_version: str = "3.1"
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0.0)
    frame_count: int = Field(gt=0)
    frames: tuple[Sam3FrameRecord, ...]

    @model_validator(mode="after")
    def validate_frames(self) -> Sam3RecordedOutput:
        indices = [frame.frame_index for frame in self.frames]
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise ValueError("frame records must have unique, increasing indices")
        if indices and indices[-1] >= self.frame_count:
            raise ValueError("frame index must be lower than frame_count")
        return self

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps


class Sam3Adapter:
    def __init__(self, *, min_confidence: float = 0.5) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in the inclusive [0, 1] range")
        self.min_confidence = min_confidence

    def convert(self, recorded: Sam3RecordedOutput) -> Scene:
        observations_by_id: dict[int, list[Observation]] = {}
        for frame in recorded.frames:
            for index, object_id in enumerate(frame.out_obj_ids):
                probability = frame.out_probs[index]
                observation = Observation(
                    frame_index=frame.frame_index,
                    timestamp_s=frame.frame_index / recorded.fps,
                    bbox=frame.out_boxes_xywh[index].to_pixel_box(
                        image_width=recorded.width,
                        image_height=recorded.height,
                    ),
                    confidence=probability,
                    visible=probability >= self.min_confidence,
                    mask_uri=frame.mask_uri_at(index),
                )
                observations_by_id.setdefault(object_id, []).append(observation)

        tracks = tuple(
            Track(
                track_id=f"sam3:{object_id}",
                concept=recorded.prompt,
                observations=tuple(observations),
            )
            for object_id, observations in sorted(observations_by_id.items())
            if any(observation.visible for observation in observations)
        )

        return Scene(
            metadata=SceneMetadata(
                scene_id=recorded.scene_id,
                source_uri=recorded.source_uri,
                width=recorded.width,
                height=recorded.height,
                fps=recorded.fps,
                duration_s=recorded.duration_s,
            ),
            tracks=tracks,
            provenance=(
                InferenceProvenance(
                    model_id=recorded.model_id,
                    model_version=recorded.model_version,
                    task="promptable concept segmentation and tracking",
                    prompt=recorded.prompt,
                ),
            ),
        )


def load_recorded_output(path: Path) -> Sam3RecordedOutput:
    return Sam3RecordedOutput.model_validate_json(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert recorded SAM 3.1 frame outputs into a Visual Evidence Graph"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args()

    recorded = load_recorded_output(args.input)
    scene = Sam3Adapter(min_confidence=args.min_confidence).convert(recorded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(scene.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

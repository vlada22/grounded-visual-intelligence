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


class GroundedObjectSeed(StrictModel):
    object_id: int = Field(gt=0)
    label: str = Field(min_length=1)
    grounding_score: float = Field(ge=0.0, le=1.0)
    prompt_frame_index: int = Field(ge=0)
    initial_box: BoundingBox


class GroundedSam2FrameRecord(StrictModel):
    """Portable output for one SAM 2.1 propagation frame."""

    frame_index: int = Field(ge=0)
    object_ids: tuple[int, ...]
    boxes_xyxy: tuple[BoundingBox, ...]
    mask_areas: tuple[int, ...]
    mask_uris: tuple[str, ...]

    @model_validator(mode="after")
    def validate_columns(self) -> GroundedSam2FrameRecord:
        expected = len(self.object_ids)
        columns = (self.boxes_xyxy, self.mask_areas, self.mask_uris)
        if any(len(column) != expected for column in columns):
            raise ValueError("Grounded SAM 2 frame columns must have equal lengths")
        if len(self.object_ids) != len(set(self.object_ids)):
            raise ValueError("object IDs must be unique within a frame")
        if any(area <= 0 for area in self.mask_areas):
            raise ValueError("recorded masks must have positive area")
        return self


class GroundedSam2RecordedOutput(StrictModel):
    schema_version: str = "1.0"
    scene_id: str = Field(min_length=1)
    source_uri: str
    prompt: str = Field(min_length=1)
    detector_model_id: str = "IDEA-Research/GroundingDINO"
    detector_model_version: str = "1.0-swin-t"
    tracker_model_id: str = "facebook/sam2"
    tracker_model_version: str = "2.1-hiera-large"
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0.0)
    frame_count: int = Field(gt=0)
    seeds: tuple[GroundedObjectSeed, ...]
    frames: tuple[GroundedSam2FrameRecord, ...]

    @model_validator(mode="after")
    def validate_records(self) -> GroundedSam2RecordedOutput:
        seed_ids = [seed.object_id for seed in self.seeds]
        if len(seed_ids) != len(set(seed_ids)):
            raise ValueError("seed object IDs must be unique")
        if any(seed.prompt_frame_index >= self.frame_count for seed in self.seeds):
            raise ValueError("prompt frame index must be lower than frame_count")

        frame_indices = [frame.frame_index for frame in self.frames]
        if frame_indices != sorted(frame_indices) or len(frame_indices) != len(set(frame_indices)):
            raise ValueError("frame records must have unique, increasing indices")
        if frame_indices and frame_indices[-1] >= self.frame_count:
            raise ValueError("frame index must be lower than frame_count")

        known_ids = set(seed_ids)
        if any(
            object_id not in known_ids for frame in self.frames for object_id in frame.object_ids
        ):
            raise ValueError("frame object IDs must reference declared seeds")
        return self

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps


class GroundedSam2Adapter:
    def convert(self, recorded: GroundedSam2RecordedOutput) -> Scene:
        seeds = {seed.object_id: seed for seed in recorded.seeds}
        observations_by_id: dict[int, list[Observation]] = {}
        for frame in recorded.frames:
            for index, object_id in enumerate(frame.object_ids):
                seed = seeds[object_id]
                observations_by_id.setdefault(object_id, []).append(
                    Observation(
                        frame_index=frame.frame_index,
                        timestamp_s=frame.frame_index / recorded.fps,
                        bbox=frame.boxes_xyxy[index],
                        confidence=seed.grounding_score,
                        mask_uri=frame.mask_uris[index],
                    )
                )

        tracks = tuple(
            Track(
                track_id=f"grounded-sam2:{object_id}",
                concept=seeds[object_id].label,
                observations=tuple(observations),
            )
            for object_id, observations in sorted(observations_by_id.items())
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
                    model_id=recorded.detector_model_id,
                    model_version=recorded.detector_model_version,
                    task="open-vocabulary box grounding",
                    prompt=recorded.prompt,
                ),
                InferenceProvenance(
                    model_id=recorded.tracker_model_id,
                    model_version=recorded.tracker_model_version,
                    task="prompted video segmentation and tracking",
                    prompt=None,
                ),
            ),
        )


def load_grounded_sam2_output(path: Path) -> GroundedSam2RecordedOutput:
    return GroundedSam2RecordedOutput.model_validate_json(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Grounding DINO + SAM 2.1 output into a Visual Evidence Graph"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    scene = GroundedSam2Adapter().convert(load_grounded_sam2_output(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(scene.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

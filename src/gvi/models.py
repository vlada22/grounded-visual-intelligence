from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Point2D(StrictModel):
    """A normalized image point in the inclusive [0, 1] range."""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class BoundingBox(StrictModel):
    """An axis-aligned bounding box in pixel coordinates."""

    x_min: float = Field(ge=0.0)
    y_min: float = Field(ge=0.0)
    x_max: float = Field(ge=0.0)
    y_max: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_extent(self) -> BoundingBox:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("bounding box must have positive width and height")
        return self

    def normalized_centroid(self, *, width: int, height: int) -> Point2D:
        return Point2D(
            x=((self.x_min + self.x_max) / 2.0) / width,
            y=((self.y_min + self.y_max) / 2.0) / height,
        )


class TimeRange(StrictModel):
    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_order(self) -> TimeRange:
        if self.end_s < self.start_s:
            raise ValueError("end_s must be greater than or equal to start_s")
        return self

    def contains(self, timestamp_s: float) -> bool:
        return self.start_s <= timestamp_s <= self.end_s


class Observation(StrictModel):
    frame_index: int = Field(ge=0)
    timestamp_s: float = Field(ge=0.0)
    bbox: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    visible: bool = True
    mask_uri: str | None = None


class Track(StrictModel):
    track_id: str = Field(min_length=1)
    concept: str = Field(min_length=1)
    observations: tuple[Observation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_observation_order(self) -> Track:
        frame_indices = [observation.frame_index for observation in self.observations]
        timestamps = [observation.timestamp_s for observation in self.observations]
        if frame_indices != sorted(frame_indices) or len(frame_indices) != len(set(frame_indices)):
            raise ValueError("track observations must have unique, increasing frame indices")
        if timestamps != sorted(timestamps):
            raise ValueError("track observations must have increasing timestamps")
        return self


class Zone(StrictModel):
    """A normalized axis-aligned region used for deterministic spatial events."""

    zone_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    x_min: float = Field(ge=0.0, le=1.0)
    y_min: float = Field(ge=0.0, le=1.0)
    x_max: float = Field(ge=0.0, le=1.0)
    y_max: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_extent(self) -> Zone:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("zone must have positive width and height")
        return self

    def contains(self, point: Point2D) -> bool:
        return self.x_min <= point.x <= self.x_max and self.y_min <= point.y <= self.y_max


class SceneMetadata(StrictModel):
    scene_id: str = Field(min_length=1)
    source_uri: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0.0)
    duration_s: float = Field(gt=0.0)


class InferenceProvenance(StrictModel):
    model_id: str = Field(min_length=1)
    model_version: str | None = None
    task: str = Field(min_length=1)
    prompt: str | None = None


class Scene(StrictModel):
    schema_version: str = "1.0"
    metadata: SceneMetadata
    tracks: tuple[Track, ...]
    zones: tuple[Zone, ...] = ()
    provenance: tuple[InferenceProvenance, ...] = ()

    @model_validator(mode="after")
    def validate_unique_ids(self) -> Scene:
        track_ids = [track.track_id for track in self.tracks]
        zone_ids = [zone.zone_id for zone in self.zones]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("scene track IDs must be unique")
        if len(zone_ids) != len(set(zone_ids)):
            raise ValueError("scene zone IDs must be unique")
        return self


class EvidenceRef(StrictModel):
    scene_id: str
    track_ids: tuple[str, ...]
    frame_indices: tuple[int, ...]
    time_range: TimeRange
    description: str


class ZoneVisit(StrictModel):
    track_id: str
    zone_id: str
    time_range: TimeRange
    observed_duration_s: float = Field(ge=0.0)
    evidence: EvidenceRef

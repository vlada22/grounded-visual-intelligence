from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from gvi.models import TimeRange
from gvi.scene_memory import SceneMemory


class FindTracksInput(BaseModel):
    concept: str = Field(description="Open-vocabulary object concept to find")
    time_range: TimeRange | None = None


class TrackInput(BaseModel):
    track_id: str


class ZoneVisitInput(BaseModel):
    track_id: str
    zone_id: str


class ToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class BoundTool:
    def __init__(
        self,
        definition: ToolDefinition,
        input_model: type[BaseModel],
        handler: Callable[[BaseModel], Any],
    ) -> None:
        self.definition = definition
        self.input_model = input_model
        self.handler = handler

    def invoke(self, arguments: dict[str, Any]) -> Any:
        return self.handler(self.input_model.model_validate(arguments))


def build_tools(memory: SceneMemory) -> tuple[BoundTool, ...]:
    return (
        _bind(
            "find_tracks",
            "Find tracked objects matching a concept, optionally during a time range.",
            FindTracksInput,
            lambda value: memory.find_tracks(value.concept, value.time_range),
        ),
        _bind(
            "count_unique",
            "Count unique tracks matching a concept, optionally during a time range.",
            FindTracksInput,
            lambda value: memory.count_unique(value.concept, value.time_range),
        ),
        _bind(
            "first_seen",
            "Return timestamped evidence for the first visible observation of a track.",
            TrackInput,
            lambda value: memory.first_seen(value.track_id),
        ),
        _bind(
            "last_seen",
            "Return timestamped evidence for the last visible observation of a track.",
            TrackInput,
            lambda value: memory.last_seen(value.track_id),
        ),
        _bind(
            "zone_visits",
            "Return evidence-backed intervals when a track was observed inside a zone.",
            ZoneVisitInput,
            lambda value: memory.zone_visits(value.track_id, value.zone_id),
        ),
        _bind(
            "observed_dwell_time",
            "Measure observed time inside a zone without interpolating unobserved frames.",
            ZoneVisitInput,
            lambda value: memory.observed_dwell_time(value.track_id, value.zone_id),
        ),
    )


def tool_manifest(tools: tuple[BoundTool, ...]) -> list[dict[str, Any]]:
    return [tool.definition.model_dump(mode="json") for tool in tools]


def _bind(
    name: str,
    description: str,
    input_model: type[BaseModel],
    handler: Callable[[Any], Any],
) -> BoundTool:
    definition = ToolDefinition(
        name=name,
        description=description,
        input_schema=input_model.model_json_schema(),
    )
    return BoundTool(definition, input_model, handler)


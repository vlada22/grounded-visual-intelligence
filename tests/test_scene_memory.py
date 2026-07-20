import pytest

from gvi.demo import build_demo_scene
from gvi.models import TimeRange
from gvi.scene_memory import SceneMemory
from gvi.tool_contracts import build_tools, tool_manifest


@pytest.fixture
def memory() -> SceneMemory:
    return SceneMemory(build_demo_scene())


def test_finds_and_counts_tracks(memory: SceneMemory) -> None:
    tracks = memory.find_tracks("toy car")

    assert [track.track_id for track in tracks] == ["car-1"]
    assert memory.count_unique("cup") == 1
    assert memory.count_unique("car", TimeRange(start_s=2.0, end_s=2.5)) == 1
    assert memory.count_unique("car", TimeRange(start_s=2.6, end_s=3.0)) == 0


def test_returns_evidence_for_first_and_last_observation(memory: SceneMemory) -> None:
    first = memory.first_seen("car-1")
    last = memory.last_seen("car-1")

    assert first.frame_indices == (0,)
    assert first.time_range.start_s == 0.0
    assert last.frame_indices == (5,)
    assert last.time_range.end_s == 2.5


def test_measures_observed_zone_visits(memory: SceneMemory) -> None:
    visits = memory.zone_visits("car-1", "center")

    assert len(visits) == 1
    assert visits[0].evidence.frame_indices == (2, 3)
    assert visits[0].observed_duration_s == pytest.approx(1.0)
    assert memory.observed_dwell_time("car-1", "center") == pytest.approx(1.0)


def test_exports_provider_neutral_tool_manifest(memory: SceneMemory) -> None:
    manifest = tool_manifest(build_tools(memory))

    assert {tool["name"] for tool in manifest} == {
        "find_tracks",
        "count_unique",
        "first_seen",
        "last_seen",
        "zone_visits",
        "observed_dwell_time",
    }
    assert all("input_schema" in tool for tool in manifest)

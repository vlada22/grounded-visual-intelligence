from __future__ import annotations

import argparse
import json
from pathlib import Path

from gvi.models import BoundingBox, Observation, Scene, SceneMetadata, Track, Zone
from gvi.scene_memory import SceneMemory
from gvi.tool_contracts import build_tools, tool_manifest


def build_demo_scene() -> Scene:
    return Scene(
        metadata=SceneMetadata(
            scene_id="tabletop-001",
            source_uri="prepared/tabletop-001.mp4",
            width=640,
            height=360,
            fps=2.0,
            duration_s=3.0,
        ),
        tracks=(
            Track(
                track_id="car-1",
                concept="red toy car",
                observations=tuple(
                    Observation(
                        frame_index=index,
                        timestamp_s=index * 0.5,
                        bbox=BoundingBox(
                            x_min=x,
                            y_min=150,
                            x_max=x + 70,
                            y_max=210,
                        ),
                        confidence=0.96 - (index * 0.01),
                    )
                    for index, x in enumerate((40, 120, 220, 310, 410, 510))
                ),
            ),
            Track(
                track_id="cup-1",
                concept="blue cup",
                observations=tuple(
                    Observation(
                        frame_index=index,
                        timestamp_s=index * 0.5,
                        bbox=BoundingBox(x_min=285, y_min=100, x_max=350, y_max=220),
                        confidence=0.98,
                    )
                    for index in range(6)
                ),
            ),
        ),
        zones=(
            Zone(
                zone_id="center",
                label="centre of the table",
                x_min=0.35,
                y_min=0.25,
                x_max=0.65,
                y_max=0.75,
            ),
        ),
    )


def write_plot(scene: Scene, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 2.8), constrained_layout=True)
    for lane, track in enumerate(scene.tracks):
        timestamps = [observation.timestamp_s for observation in track.observations]
        axis.scatter(timestamps, [lane] * len(timestamps), s=90, label=track.concept)
        axis.plot(timestamps, [lane] * len(timestamps), alpha=0.55)
    axis.set(
        title="Visual evidence occupancy",
        xlabel="Time (seconds)",
        yticks=range(len(scene.tracks)),
        yticklabels=[track.concept for track in scene.tracks],
    )
    axis.grid(axis="x", alpha=0.25)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic visual evidence example")
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    scene = build_demo_scene()
    memory = SceneMemory(scene)
    tools = build_tools(memory)

    (args.output / "scene.json").write_text(
        scene.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (args.output / "tools.json").write_text(
        json.dumps(tool_manifest(tools), indent=2),
        encoding="utf-8",
    )
    write_plot(scene, args.output / "track-occupancy.png")

    summary = {
        "red_car_count": memory.count_unique("toy car"),
        "car_first_seen": memory.first_seen("car-1").model_dump(mode="json"),
        "car_center_visits": [
            visit.model_dump(mode="json") for visit in memory.zone_visits("car-1", "center")
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

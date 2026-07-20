from __future__ import annotations

import csv
import hashlib
import json
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from gvi.inference.mask_rle import BinaryMaskRle, decode_binary_mask
from gvi.models import Scene
from gvi.scene_memory import SceneMemory


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def runtime_metadata(torch_module: Any) -> dict[str, Any]:
    device = torch_module.cuda.get_device_properties(0)
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch_module.__version__,
        "cuda_version": torch_module.version.cuda,
        "cudnn_version": torch_module.backends.cudnn.version(),
        "gpu": device.name,
        "gpu_compute_capability": f"{device.major}.{device.minor}",
        "gpu_total_memory_gib": device.total_memory / 1024**3,
        "peak_process_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "packages": {
            name: package_version(name)
            for name in (
                "accelerate",
                "av",
                "groundingdino",
                "huggingface-hub",
                "numpy",
                "opencv-python",
                "pillow",
                "sam-2",
                "transformers",
            )
        },
    }


def _load_mask(output_dir: Path, mask_uri: str | None) -> np.ndarray | None:
    if not mask_uri:
        return None
    encoded = BinaryMaskRle.model_validate_json(
        (output_dir / mask_uri).read_text(encoding="utf-8")
    )
    return decode_binary_mask(encoded)


def build_analysis_artifacts(
    *,
    output_dir: Path,
    video_path: Path,
    scene_path: Path,
) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    prepared_path = output_dir / "prepared-input.mp4"
    shutil.copy2(video_path, prepared_path)

    observations_by_frame: dict[int, list[tuple[str, str, dict[str, Any]]]] = {}
    rows: list[dict[str, Any]] = []
    previous_masks: dict[str, np.ndarray] = {}
    track_frames: dict[str, list[int]] = {}

    for track in scene["tracks"]:
        track_id = track["track_id"]
        concept = track["concept"]
        frames = []
        for observation in track["observations"]:
            frame_index = int(observation["frame_index"])
            frames.append(frame_index)
            observations_by_frame.setdefault(frame_index, []).append(
                (track_id, concept, observation)
            )
            mask = _load_mask(output_dir, observation.get("mask_uri"))
            bbox = observation["bbox"]
            row: dict[str, Any] = {
                "track_id": track_id,
                "concept": concept,
                "frame_index": frame_index,
                "timestamp_s": observation["timestamp_s"],
                "confidence": observation["confidence"],
                "visible": observation["visible"],
                "bbox_x_min": bbox["x_min"],
                "bbox_y_min": bbox["y_min"],
                "bbox_x_max": bbox["x_max"],
                "bbox_y_max": bbox["y_max"],
                "bbox_area_px": (bbox["x_max"] - bbox["x_min"])
                * (bbox["y_max"] - bbox["y_min"]),
                "mask_uri": observation.get("mask_uri"),
                "mask_area_px": None,
                "mask_fraction": None,
                "mask_centroid_x": None,
                "mask_centroid_y": None,
                "mask_iou_previous": None,
            }
            if mask is not None:
                y_coords, x_coords = np.nonzero(mask)
                area = int(mask.sum())
                row["mask_area_px"] = area
                row["mask_fraction"] = area / mask.size
                if area:
                    row["mask_centroid_x"] = float(x_coords.mean())
                    row["mask_centroid_y"] = float(y_coords.mean())
                previous = previous_masks.get(track_id)
                if previous is not None:
                    union = np.logical_or(previous, mask).sum()
                    row["mask_iou_previous"] = (
                        float(np.logical_and(previous, mask).sum() / union) if union else 1.0
                    )
                previous_masks[track_id] = mask
            rows.append(row)
        track_frames[track_id] = frames

    rows.sort(key=lambda row: (row["frame_index"], row["track_id"]))
    measurements_path = output_dir / "track-measurements.csv"
    with measurements_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    frame_count = int(scene["metadata"]["duration_s"] * scene["metadata"]["fps"])
    objects_per_frame = [
        len(observations_by_frame.get(frame_index, [])) for frame_index in range(frame_count)
    ]
    confidences = [float(row["confidence"]) for row in rows]
    mask_ious = [
        float(row["mask_iou_previous"])
        for row in rows
        if row["mask_iou_previous"] is not None
    ]
    scene_model = Scene.model_validate(scene)
    memory = SceneMemory(scene_model)
    zone_visits = {
        track.track_id: {
            zone.zone_id: [
                visit.model_dump(mode="json")
                for visit in memory.zone_visits(track.track_id, zone.zone_id)
            ]
            for zone in scene_model.zones
        }
        for track in scene_model.tracks
    }

    summary = {
        "track_count": len(scene["tracks"]),
        "observation_count": len(rows),
        "visible_observation_count": sum(bool(row["visible"]) for row in rows),
        "frames_with_evidence": sum(count > 0 for count in objects_per_frame),
        "source_frame_count": frame_count,
        "objects_per_frame": objects_per_frame,
        "track_frames": track_frames,
        "zone_visits": zone_visits,
        "track_gap_counts": {
            track_id: sum(
                right - left - 1
                for left, right in zip(frames, frames[1:], strict=False)
                if right > left + 1
            )
            for track_id, frames in track_frames.items()
        },
        "confidence": {
            "min": min(confidences) if confidences else None,
            "max": max(confidences) if confidences else None,
            "mean": float(np.mean(confidences)) if confidences else None,
            "unique_count": len(set(confidences)),
        },
        "mask_iou_previous": {
            "min": min(mask_ious) if mask_ious else None,
            "max": max(mask_ious) if mask_ious else None,
            "mean": float(np.mean(mask_ious)) if mask_ious else None,
        },
    }
    (output_dir / "analysis-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    colors = (
        (255, 86, 86),
        (72, 187, 255),
        (110, 231, 183),
        (251, 191, 36),
        (196, 181, 253),
    )
    with tempfile.TemporaryDirectory(prefix="gvi-preview-") as temporary:
        temp_dir = Path(temporary)
        raw_dir = temp_dir / "raw"
        annotated_dir = temp_dir / "annotated"
        raw_dir.mkdir()
        annotated_dir.mkdir()
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-vsync",
                "0",
                "-start_number",
                "0",
                str(raw_dir / "frame-%05d.png"),
            ],
            check=True,
        )
        annotated_paths = []
        for frame_path in sorted(raw_dir.glob("frame-*.png")):
            frame_index = int(frame_path.stem.split("-")[1])
            image = Image.open(frame_path).convert("RGB")
            pixels = np.asarray(image).copy()
            frame_observations = observations_by_frame.get(frame_index, [])
            for index, (track_id, concept, observation) in enumerate(frame_observations):
                color = colors[index % len(colors)]
                mask = _load_mask(output_dir, observation.get("mask_uri"))
                if mask is not None:
                    if mask.shape != pixels.shape[:2]:
                        mask = np.asarray(
                            Image.fromarray(mask).resize(
                                (pixels.shape[1], pixels.shape[0]), Image.Resampling.NEAREST
                            ),
                            dtype=bool,
                        )
                    pixels[mask] = (
                        pixels[mask].astype(np.float32) * 0.55
                        + np.asarray(color, dtype=np.float32) * 0.45
                    ).astype(np.uint8)
            image = Image.fromarray(pixels)
            draw = ImageDraw.Draw(image)
            for zone in scene.get("zones", []):
                zone_box = (
                    round(zone["x_min"] * image.width),
                    round(zone["y_min"] * image.height),
                    round(zone["x_max"] * image.width),
                    round(zone["y_max"] * image.height),
                )
                draw.rectangle(zone_box, outline=(255, 255, 255), width=2)
                draw.text(
                    (zone_box[0] + 6, zone_box[1] + 6),
                    f"{zone['label']} ({zone['zone_id']})",
                    fill=(255, 255, 255),
                )
            for index, (track_id, concept, observation) in enumerate(frame_observations):
                color = colors[index % len(colors)]
                bbox = observation["bbox"]
                draw.rectangle(
                    (
                        bbox["x_min"],
                        bbox["y_min"],
                        bbox["x_max"],
                        bbox["y_max"],
                    ),
                    outline=color,
                    width=3,
                )
                label = f"{track_id} {concept} {observation['confidence']:.3f}"
                draw.text((bbox["x_min"] + 3, max(2, bbox["y_min"] - 14)), label, fill=color)
            annotated_path = annotated_dir / frame_path.name
            image.save(annotated_path)
            annotated_paths.append(annotated_path)

        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(scene["metadata"]["fps"]),
                "-start_number",
                "0",
                "-i",
                str(annotated_dir / "frame-%05d.png"),
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output_dir / "overlay-preview.mp4"),
            ],
            check=True,
        )

        if annotated_paths:
            columns = min(4, len(annotated_paths))
            rows_count = (len(annotated_paths) + columns - 1) // columns
            thumb_width = 320
            first = Image.open(annotated_paths[0]).convert("RGB")
            thumb_height = round(first.height * thumb_width / first.width)
            sheet = Image.new(
                "RGB",
                (columns * thumb_width, rows_count * thumb_height),
                color=(20, 24, 32),
            )
            for index, frame_path in enumerate(annotated_paths):
                thumb = Image.open(frame_path).convert("RGB")
                thumb.thumbnail((thumb_width, thumb_height))
                sheet.paste(
                    thumb,
                    ((index % columns) * thumb_width, (index // columns) * thumb_height),
                )
            sheet.save(output_dir / "contact-sheet.jpg", quality=90)

    return summary


def write_manifest(output_dir: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact-manifest.json":
            entries.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = {"schema_version": "1.0", "files": entries}
    (output_dir / "artifact-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest

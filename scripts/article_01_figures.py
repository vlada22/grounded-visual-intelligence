from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont


SAM2 = Path()
SAM3 = Path()
OUT = Path()

INK = "#17212B"
MUTED = "#64748B"
GRID = "#D9E2EC"
SAM3_COLOR = "#0F8B8D"
SAM2_COLOR = "#E76F51"
ACCENT = "#F4A261"
PAPER = "#FCFCF8"


def configure() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "svg.fonttype": "none",
        }
    )


def materialize(source: Path, target: Path) -> Path:
    """Return an artifact directory, safely extracting a notebook ZIP if needed."""
    if source.is_dir():
        return source
    if not source.is_file():
        raise FileNotFoundError(source)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe archive member: {member.filename}")
        archive.extractall(target)
    return target


def decode_rle(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    counts = payload["counts"]
    values = np.arange(len(counts), dtype=np.uint8) % 2
    flat = np.repeat(values, counts)
    expected = payload["height"] * payload["width"]
    if flat.size != expected:
        raise ValueError(f"invalid RLE length in {path}: {flat.size} != {expected}")
    return flat.reshape(payload["height"], payload["width"]).astype(bool)


def mask_series() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ious, bbox_ious, centroids = [], [], []
    scenes = [json.loads((root / "scene.json").read_text()) for root in (SAM2, SAM3)]
    observations = [scene["tracks"][0]["observations"] for scene in scenes]
    for frame in range(40):
        a = decode_rle(SAM2 / "masks" / f"frame-{frame:05d}-object-1.rle.json")
        b = decode_rle(SAM3 / "masks" / f"frame-{frame:05d}-object-0.rle.json")
        ious.append(np.logical_and(a, b).sum() / np.logical_or(a, b).sum())

        cents = []
        for mask in (a, b):
            ys, xs = np.nonzero(mask)
            cents.append((xs.mean(), ys.mean()))
        boxes = []
        for observation_set in observations:
            bbox = observation_set[frame]["bbox"]
            boxes.append((bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]))
        x1 = max(boxes[0][0], boxes[1][0])
        y1 = max(boxes[0][1], boxes[1][1])
        x2 = min(boxes[0][2], boxes[1][2])
        y2 = min(boxes[0][3], boxes[1][3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        areas = [(x2_ - x1_) * (y2_ - y1_) for x1_, y1_, x2_, y2_ in boxes]
        bbox_ious.append(intersection / (areas[0] + areas[1] - intersection))
        centroids.append(np.linalg.norm(np.subtract(cents[0], cents[1])))
    return np.array(ious), np.array(bbox_ious), np.array(centroids)


def save_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        (0.35, 1.7, 2.0, 1.45, "Video", "immutable input", "#E8EEF4"),
        (3.0, 2.35, 2.25, 1.45, "Perception", "SAM 3 or\nGrounded SAM 2", "#D9F0EF"),
        (5.95, 1.7, 2.25, 1.45, "Evidence graph", "tracks · masks ·\ntime · provenance", "#FFF0DB"),
        (8.9, 2.35, 2.5, 1.45, "Deterministic tools", "count · dwell ·\ntransition · retrieve", "#E9E5F7"),
        (8.9, 0.45, 2.5, 1.45, "LLM", "plan · call tools ·\nexplain with citations", "#E8EEF4"),
    ]
    for x, y, w, h, title, subtitle, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                linewidth=1.2, edgecolor="#B7C2CC", facecolor=color,
            )
        )
        ax.text(x + 0.16, y + h - 0.38, title, color=INK, weight="bold", fontsize=12)
        ax.text(x + 0.16, y + h - 0.75, subtitle, color=MUTED, fontsize=9.5, va="top")

    arrows = [
        ((2.35, 2.43), (3.0, 3.08)),
        ((5.25, 3.08), (5.95, 2.43)),
        ((8.2, 2.43), (8.9, 3.08)),
        ((10.15, 2.35), (10.15, 1.9)),
        ((8.9, 1.18), (8.2, 2.05)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                     linewidth=1.5, color="#718096"))

    ax.text(6.0, 4.7, "Measure first. Generate language second.", fontsize=18,
            weight="bold", color=INK, ha="center")
    ax.text(6.0, 4.34, "The LLM never has to invent a timestamp that the evidence layer can compute.",
            fontsize=10.5, color=MUTED, ha="center")
    fig.savefig(OUT / "01-evidence-pipeline.svg", bbox_inches="tight")
    fig.savefig(OUT / "01-evidence-pipeline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_agreement() -> None:
    ious, bbox_ious, centroids = mask_series()
    frames = np.arange(40)
    seconds = frames / 4
    minimum = int(np.argmin(ious))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(seconds, ious, color=SAM3_COLOR, linewidth=2.4, label="Mask IoU")
    ax1.plot(seconds, bbox_ious, color=SAM2_COLOR, linewidth=1.8, alpha=0.85,
             label="Bounding-box IoU")
    ax1.axvspan(3.75, 4.0, color=ACCENT, alpha=0.18)
    ax1.scatter(seconds[minimum], ious[minimum], s=55, color=SAM2_COLOR, zorder=3)
    ax1.annotate(
        f"minimum mask IoU {ious[minimum]:.4f}\nframe {minimum} · {seconds[minimum]:.2f} s",
        xy=(seconds[minimum], ious[minimum]), xytext=(5.0, 0.951),
        arrowprops={"arrowstyle": "->", "color": MUTED}, fontsize=9.5, color=INK,
    )
    ax1.set_ylim(0.94, 0.99)
    ax1.set_ylabel("Cross-model overlap")
    fig.suptitle("Two perception stacks produced nearly identical spatial evidence",
                 fontsize=16, weight="bold", color=INK, y=0.985)
    fig.text(0.5, 0.94, "40 matched observations · one cup track · transition window shaded",
             color=MUTED, fontsize=10, ha="center")
    ax1.legend(frameon=False, loc="lower right", ncol=2)
    ax1.grid(axis="y", color=GRID, linewidth=0.8)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(seconds, centroids, color="#6C5CE7", linewidth=2)
    ax2.fill_between(seconds, centroids, color="#6C5CE7", alpha=0.12)
    ax2.axvspan(3.75, 4.0, color=ACCENT, alpha=0.18)
    ax2.axhline(centroids.mean(), color=MUTED, linestyle="--", linewidth=1)
    ax2.text(9.75, centroids.mean() + 0.05, f"mean {centroids.mean():.3f} px",
             ha="right", color=MUTED, fontsize=9)
    ax2.set_ylabel("Centroid distance (px)")
    ax2.set_xlabel("Video time (seconds)")
    ax2.set_ylim(0, 1.2)
    ax2.grid(axis="y", color=GRID, linewidth=0.8)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(OUT / "03-cross-model-agreement.svg", bbox_inches="tight")
    fig.savefig(OUT / "03-cross-model-agreement.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "mask_iou_mean": float(ious.mean()),
        "mask_iou_median": float(np.median(ious)),
        "mask_iou_min": float(ious.min()),
        "mask_iou_min_frame": minimum,
        "mask_iou_max": float(ious.max()),
        "bbox_iou_mean": float(bbox_ious.mean()),
        "centroid_distance_mean_px": float(centroids.mean()),
        "centroid_distance_max_px": float(centroids.max()),
    }
    (OUT / "computed-summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def save_profile() -> None:
    labels = ["Model load", "Prompt / grounding", "Session init", "Propagation"]
    sam2 = np.array([15.718952628, 4.025560835, 1.600828564, 15.957526364])
    sam3 = np.array([8.868453401, 0.001386285, 0.511374125, 10.185811792])
    memory_labels = ["GPU allocated", "GPU reserved", "Process RSS"]
    mem2 = np.array([3.282984257, 6.2421875, 3.875968933])
    mem3 = np.array([1.809809208, 1.876953125, 5.23285675])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.3))
    y = np.arange(len(labels))
    h = 0.34
    ax1.barh(y + h / 2, sam2, height=h, color=SAM2_COLOR, label="Grounded SAM 2")
    ax1.barh(y - h / 2, sam3, height=h, color=SAM3_COLOR, label="SAM 3")
    ax1.set_yticks(y, labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("Recorded seconds")
    ax1.set_title("Model-path phases")
    ax1.grid(axis="x", color=GRID, linewidth=0.8)
    ax1.spines[["top", "right", "left"]].set_visible(False)
    for i, values in enumerate((sam2, sam3)):
        offset = h / 2 if i == 0 else -h / 2
        for j, value in enumerate(values):
            if value >= 0.1:
                ax1.text(value + 0.2, j + offset, f"{value:.1f}", va="center", fontsize=8.5)

    x = np.arange(len(memory_labels))
    ax2.bar(x - 0.18, mem2, width=0.36, color=SAM2_COLOR, label="Grounded SAM 2")
    ax2.bar(x + 0.18, mem3, width=0.36, color=SAM3_COLOR, label="SAM 3")
    ax2.set_xticks(x, ["GPU\nallocated", "GPU\nreserved", "Process\nRSS"])
    ax2.set_ylabel("GiB")
    ax2.set_title("Peak memory observed")
    ax2.grid(axis="y", color=GRID, linewidth=0.8)
    ax2.spines[["top", "right"]].set_visible(False)
    for xi, value in zip(x - 0.18, mem2, strict=True):
        ax2.text(xi, value + 0.12, f"{value:.2f}", ha="center", fontsize=8.5)
    for xi, value in zip(x + 0.18, mem3, strict=True):
        ax2.text(xi, value + 0.12, f"{value:.2f}", ha="center", fontsize=8.5)

    fig.suptitle("Recorded operational profile on one Colab T4 run", x=0.5, y=0.98,
                 fontsize=16, weight="bold", color=INK)
    handles, legend_labels = ax1.get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 0.91), ncol=2)
    fig.text(0.5, -0.01,
             "Run provenance, not a general performance benchmark; downloads excluded from phase chart.",
             ha="center", color=MUTED, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.84))
    fig.savefig(OUT / "04-run-profile.svg", bbox_inches="tight")
    fig.savefig(OUT / "04-run-profile.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def extract_frame(video: Path, seconds: float, target: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(seconds), "-i", str(video),
         "-frames:v", "1", "-y", str(target)], check=True
    )


def save_transition_montage() -> None:
    times = [3.25, 3.75, 4.00, 4.50]
    labels = ["Approaching boundary", "Last evidence in zone A", "First evidence in zone B",
              "Established in zone B"]
    images = []
    for i, seconds in enumerate(times):
        target = OUT / f"frame-{i}.png"
        extract_frame(SAM3 / "overlay-preview.mp4", seconds, target)
        images.append(Image.open(target).convert("RGB"))

    width, height = images[0].size
    header = 74
    gap = 12
    canvas = Image.new("RGB", (width * 2 + gap * 3, (height + header) * 2 + gap * 3), PAPER)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    for i, (img, seconds, label) in enumerate(zip(images, times, labels, strict=True)):
        col, row = i % 2, i // 2
        x = gap + col * (width + gap)
        y = gap + row * (height + header + gap)
        draw.rounded_rectangle((x, y, x + width, y + header - 8), radius=9, fill="#17212B")
        draw.text((x + 15, y + 10), f"{seconds:.2f} s", font=bold, fill="#FFFFFF")
        draw.text((x + 15, y + 38), label, font=small, fill="#D9E2EC")
        canvas.paste(img, (x, y + header))
    canvas.save(OUT / "02-transition-window.jpg", quality=90, optimize=True)
    for i in range(len(times)):
        (OUT / f"frame-{i}.png").unlink()


def save_gif() -> None:
    palette = OUT / "palette.png"
    output = OUT / "cup-transition-evidence.gif"
    base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", "2.75", "-t", "3.5",
            "-i", str(SAM3 / "overlay-preview.mp4")]
    vf = "fps=6,scale=520:-1:flags=lanczos"
    subprocess.run(base + ["-vf", vf + ",palettegen=max_colors=96:stats_mode=diff", "-y", str(palette)],
                   check=True)
    subprocess.run(base + ["-i", str(palette), "-lavfi", vf + " [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=4",
                           "-loop", "0", "-y", str(output)], check=True)
    palette.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the publication figures for Article 01")
    parser.add_argument("--sam2", type=Path, required=True,
                        help="Grounded SAM 2 artifact directory or ZIP")
    parser.add_argument("--sam3", type=Path, required=True,
                        help="SAM 3 artifact directory or ZIP")
    parser.add_argument("--output", type=Path, default=Path("docs/article-01/assets"))
    args = parser.parse_args()

    global SAM2, SAM3, OUT
    OUT = args.output.resolve()
    configure()
    with tempfile.TemporaryDirectory(prefix="gvi-article-01-") as temp_dir:
        temp_root = Path(temp_dir)
        SAM2 = materialize(args.sam2.resolve(), temp_root / "sam2")
        SAM3 = materialize(args.sam3.resolve(), temp_root / "sam3")
        save_pipeline()
        save_transition_montage()
        save_agreement()
        save_profile()
        save_gif()
    for path in sorted(OUT.iterdir()):
        print(f"{path.name}\t{path.stat().st_size}")


if __name__ == "__main__":
    main()

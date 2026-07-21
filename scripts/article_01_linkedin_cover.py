from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent / "linkedin-assets"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 6.44), dpi=100, facecolor="#101A24")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 644)
    ax.axis("off")

    # Quiet visual texture that survives LinkedIn's center cropping.
    for x, y, radius, color, alpha in [
        (1060, 550, 250, "#0F8B8D", 0.13),
        (1170, 110, 280, "#E76F51", 0.11),
        (870, 120, 180, "#F4A261", 0.07),
    ]:
        ax.add_patch(Circle((x, y), radius, facecolor=color, edgecolor="none", alpha=alpha))

    ax.text(72, 574, "GROUNDED VISUAL INTELLIGENCE  ·  ARTICLE 01",
            color="#7ED6D8", fontsize=14, fontweight="bold", va="center")
    ax.text(72, 468, "Ask the video,", color="#F7FAFC", fontsize=48,
            fontweight="bold", va="center")
    ax.text(72, 405, "not just the VLM.", color="#F4A261", fontsize=48,
            fontweight="bold", va="center")
    ax.text(75, 319, "Evidence-grounded visual memory with SAM 3,",
            color="#CFD8E3", fontsize=18, va="center")
    ax.text(75, 286, "Grounded SAM 2, and deterministic tools",
            color="#CFD8E3", fontsize=18, va="center")

    # Compact evidence chain on the right.
    nodes = [
        (760, 432, 118, 62, "OBSERVE", "#173C47"),
        (925, 432, 118, 62, "MEASURE", "#4B3340"),
        (1090, 432, 88, 62, "CITE", "#3C354F"),
    ]
    for x, y, width, height, label, color in nodes:
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                                   boxstyle="round,pad=0.02,rounding_size=12",
                                   facecolor=color, edgecolor="#7A8C9E", linewidth=1.2))
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center",
                color="#F7FAFC", fontsize=12, fontweight="bold")
    for start, end in [((878, 463), (925, 463)), ((1043, 463), (1090, 463))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                     color="#A7B6C5", linewidth=1.4))

    ax.plot([72, 1128], [111, 111], color="#334353", linewidth=1)
    ax.text(72, 73, "github.com/vlada22/grounded-visual-intelligence",
            color="#9FB0C1", fontsize=13, va="center")
    ax.text(1128, 73, "MEASURE FIRST  ·  GENERATE SECOND",
            color="#9FB0C1", fontsize=11, ha="right", va="center", fontweight="bold")

    output = OUT / "linkedin-cover.png"
    fig.savefig(output, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

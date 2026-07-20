# Grounded Visual Intelligence

Experiments in turning video into structured, measurable evidence that language
models can query without inventing visual facts.

The first article, **Ask the Video, Not Just the VLM**, builds a temporal visual
memory from object tracks and exposes deterministic tools for counting, timing,
zone occupancy, and evidence retrieval.

[Open the GitHub Pages evidence explorer](https://vlada22.github.io/grounded-visual-intelligence/)

## Current checkpoint

- Visual Evidence Graph domain model
- validated scene serialization
- temporal and spatial query tools
- provider-neutral LLM tool contracts
- synthetic executable example
- Grounding DINO + SAM 2.1 recorded-output adapter and fixture
- ungated Colab inference path
- official Transformers SAM 3 video comparison notebook
- model-neutral SAM 3 adapter and dependency-injected worker
- portable binary-mask RLE artifacts
- matched cup-crossing results from both perception stacks
- artifact-backed browser explorer deployed through GitHub Pages

The core package deliberately has no PyTorch or SAM dependency. GPU inference
runs behind an adapter and emits portable recorded-output artifacts.

## Run locally

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
uv run pytest
uv run gvi-demo --output outputs/demo
uv run gvi-ingest-grounded-sam2 \
  tests/fixtures/grounded_sam2_tabletop_output.json \
  outputs/grounded-sam2-tabletop-scene.json
```

The demo writes a scene artifact, an LLM tool manifest, and a track-occupancy
plot to `outputs/demo`.

## Run the browser explorer locally

The explorer is dependency-free. Serve the repository root so it can use the
versioned source video:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/web/`. GitHub Pages deployment is defined in
`.github/workflows/pages.yml`.

## Compare the perception stacks on Colab

Both GPU paths are offline batch jobs that emit the same model-independent
evidence schema:

- Open
  [`notebooks/article_01_grounded_sam2_colab.ipynb`](notebooks/article_01_grounded_sam2_colab.ipynb)
  for the public Grounding DINO 1.0 + SAM 2.1 baseline.
- Open [`notebooks/article_01_sam3_colab.ipynb`](notebooks/article_01_sam3_colab.ipynb)
  for the gated `facebook/sam3` Transformers video comparison.

Both notebooks load the approved Gemini-generated
`assets/article-01/sample.mp4` from the repository and verify its SHA-256 before
GPU work. They process the full ten-second clip with the `white cup` prompt,
4 fps, a 560-pixel maximum width, and identical A/B zones. While this repository
is private, both require a read-only `GITHUB_TOKEN`. The SAM 3 runner additionally
requires an `HF_TOKEN` approved for `facebook/sam3`, keeps decoded frames on CPU,
and selects FP16 on a T4.

[Read the validated matched-run results](docs/article-01/results.md). Both
pipelines produced one complete 40-frame cup track, identical zone measurements,
zero gaps, and 0.9694 mean cross-model mask IoU.

## Project direction

1. Evidence-grounded video memory
2. 3D geometry and spatial tools
3. Occlusion, re-identification, and uncertainty
4. VLM answers versus deterministic visual measurements
5. Robustness under real-world degradation
6. Local and browser deployment

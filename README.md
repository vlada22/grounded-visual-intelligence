# Grounded Visual Intelligence

Experiments in turning video into structured, measurable evidence that language
models can query without inventing visual facts.

The first article, **Ask the Video, Not Just the VLM**, builds a temporal visual
memory from object tracks and exposes deterministic tools for counting,
timing, zone occupancy, and evidence retrieval.

[Open the interactive evidence explorer](https://grounded-visual-intelligence-lab.vlatko-nikol-0153.chatgpt.site)

## Current checkpoint

- Visual Evidence Graph domain model
- validated scene serialization
- temporal and spatial query tools
- provider-neutral LLM tool contracts
- synthetic executable example
- initial Article 1 brief and evaluation plan
- Grounding DINO + SAM 2.1 recorded-output adapter and fixture
- ungated Colab inference path
- optional SAM 3.1 adapter and dependency-injected worker
- portable binary-mask RLE artifacts

The core package deliberately has no PyTorch or SAM dependency. GPU inference
will run behind an adapter and emit portable recorded-output artifacts.

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

## Run the perception stack on Colab

The GPU path is intentionally an offline batch job. Open
[`notebooks/article_01_grounded_sam2_colab.ipynb`](notebooks/article_01_grounded_sam2_colab.ipynb)
in Google Colab and select a GPU runtime. While this repository is private, add
a read-only `GITHUB_TOKEN` through Colab Secrets. Grounding DINO 1.0 and SAM 2.1
weights come from public release URLs; no gated model token is required. The
notebook uploads one prepared video and downloads a portable evidence bundle.

## Project direction

1. Evidence-grounded video memory
2. 3D geometry and spatial tools
3. Occlusion, re-identification, and uncertainty
4. VLM answers versus deterministic visual measurements
5. Robustness under real-world degradation
6. Local and browser deployment

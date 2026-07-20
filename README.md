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
- SAM 3.1 Object Multiplex comparison notebook
- model-neutral SAM 3.1 adapter and dependency-injected worker
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

## Compare the perception stacks on Colab

Both GPU paths are offline batch jobs that emit the same model-independent
evidence schema:

- Open
[`notebooks/article_01_grounded_sam2_colab.ipynb`](notebooks/article_01_grounded_sam2_colab.ipynb)
  for the public Grounding DINO 1.0 + SAM 2.1 baseline.
- Open [`notebooks/article_01_sam3_colab.ipynb`](notebooks/article_01_sam3_colab.ipynb)
  for the gated SAM 3.1 Object Multiplex comparison.

Use the identical source video and semantic prompt in both notebooks before
tuning either model. While this repository is private, both require a read-only
`GITHUB_TOKEN`. The SAM 3.1 runner additionally requires an `HF_TOKEN` approved
for `facebook/sam3.1` and recommends at least 20 GiB of GPU memory. The resulting
ZIP archives have distinct pipeline names and include `scene.json`, recorded
model output, masks, and run metrics.

## Project direction

1. Evidence-grounded video memory
2. 3D geometry and spatial tools
3. Occlusion, re-identification, and uncertainty
4. VLM answers versus deterministic visual measurements
5. Robustness under real-world degradation
6. Local and browser deployment

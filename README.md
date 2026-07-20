# Grounded Visual Intelligence

Experiments in turning video into structured, measurable evidence that language
models can query without inventing visual facts.

The first article, **Ask the Video, Not Just the VLM**, builds a temporal visual
memory from object tracks and exposes deterministic tools for counting,
timing, zone occupancy, and evidence retrieval.

## Current checkpoint

- Visual Evidence Graph domain model
- validated scene serialization
- temporal and spatial query tools
- provider-neutral LLM tool contracts
- synthetic executable example
- initial Article 1 brief and evaluation plan
- SAM 3.1 recorded-output adapter and fixture
- dependency-injected SAM 3.1 inference worker
- portable binary-mask RLE artifacts

The core package deliberately has no PyTorch or SAM dependency. GPU inference
will run behind an adapter and emit portable recorded-output artifacts.

## Run locally

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
uv run pytest
uv run gvi-demo --output outputs/demo
uv run gvi-ingest-sam3 \
  tests/fixtures/sam3_tabletop_output.json \
  outputs/sam3-tabletop-scene.json
```

The demo writes a scene artifact, an LLM tool manifest, and a track-occupancy
plot to `outputs/demo`.

## Project direction

1. Evidence-grounded video memory
2. 3D geometry and spatial tools
3. Occlusion, re-identification, and uncertainty
4. VLM answers versus deterministic visual measurements
5. Robustness under real-world degradation
6. Local and browser deployment

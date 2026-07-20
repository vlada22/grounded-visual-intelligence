# SAM 3 versus Grounded SAM 2 comparison

This is an applied comparison, not a general model benchmark. Its purpose is to
select the most useful perception path for Article 1 and make the choice
inspectable.

## Controlled first run

Run both Colab notebooks with:

- the same source MP4;
- the approved Gemini-generated `sample.mp4` with the expected source digest;
- the same semantic concept (`white cup`);
- prompt frame zero;
- default confidence thresholds;
- no interactive corrections or model-specific tuning.

Grounding DINO conventionally receives the concept with a trailing period, but
the recorded prompt is normalized without it. Both notebooks deterministically
prepare the full ten-second clip at 4 fps and a 560-pixel maximum width. The
Transformers SAM 3 runner keeps decoded video on CPU and uses FP16 on a T4.
Zones A and B exclude the narrow normalized x=0.49–0.51 divider band. This is an
applied correctness run, not a general throughput benchmark.

## Return artifacts

Keep the two archives separate:

- `article-01-cup-crossing-grounded-sam2.zip`
- `article-01-cup-crossing-sam3.zip`

Each archive contains:

- the deterministically prepared ten-second comparison clip;
- an annotated overlay preview and contact sheet for visual inspection;
- per-frame track measurements and an aggregate analysis summary;
- the raw model recording and model-independent `scene.json`;
- portable RLE masks;
- phase timings, GPU/runtime/package provenance, source and checkpoint hashes;
- an integrity manifest with a SHA-256 digest for every included file.

Do not include access tokens, original user-uploaded source videos, model weights,
or full model caches. The prepared clip is an internal comparison artifact; check
redistribution rights before publishing it.

## What we will inspect

1. expected instances found on the prompt frame;
2. objects discovered after the prompt frame;
3. identity switches and duplicate tracks;
4. missing-mask gaps around occlusion;
5. visibly incorrect or leaking masks;
6. whether Transformers SAM 3 completes on the 15 GiB T4 preset;
7. model load, prompt/detection, session initialization, and propagation time;
8. peak allocated and reserved GPU memory plus peak process RSS;
9. mask-area and consecutive-mask-IoU stability;
10. whether both artifacts answer the same deterministic questions correctly.

If one pipeline fails before inference, preserve the complete error text and
GPU model. Setup reliability is part of the applied result.

# SAM 3.1 versus Grounded SAM 2 comparison

This is an applied comparison, not a general model benchmark. Its purpose is to
select the most useful perception path for Article 1 and make the choice
inspectable.

## Controlled first run

Run both Colab notebooks with:

- the same source MP4;
- the same semantic concept (`red toy car` by default);
- prompt frame zero;
- default confidence thresholds;
- no interactive corrections or model-specific tuning.

Grounding DINO conventionally receives the concept with a trailing period, but
the recorded prompt is normalized without it. Both notebooks deterministically
prepare the first three seconds at 4 fps and at most 640 pixels wide. The SAM 3.1
runner also disables compilation and Flash Attention 3, uses FP16 on a T4, and
caps output at four objects. This is a correctness smoke test, not a final
throughput comparison.

## Return artifacts

Keep the two archives separate:

- `article-01-hero-grounded-sam2.zip`
- `article-01-hero-sam31.zip`

Each archive contains the raw model recording, model-independent `scene.json`,
RLE masks, and `run-metrics.json`. Do not include the source video, access
tokens, or model weights.

## What we will inspect

1. expected instances found on the prompt frame;
2. objects discovered after the prompt frame;
3. identity switches and duplicate tracks;
4. missing-mask gaps around occlusion;
5. visibly incorrect or leaking masks;
6. whether SAM 3.1 completes on the 15 GiB T4 preset;
7. smoke-test elapsed time and peak allocated GPU memory;
8. whether both artifacts answer the same deterministic questions correctly.

If one pipeline fails before inference, preserve the complete error text and
GPU model. Setup reliability is part of the applied result.

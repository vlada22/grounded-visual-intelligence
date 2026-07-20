# Article 1 — matched perception run

## Status

The first controlled run is complete. Both pipelines processed the identical
prepared clip and produced valid, complete evidence archives.

This is an applied smoke test on one short clip, not a general model benchmark.

## Reproducibility

| Field | Value |
| --- | --- |
| Runtime | Google Colab, Tesla T4 (14.56 GiB), CUDA 12.8, Torch 2.11.0 |
| Source | `hf-internal-testing/sam2-fixtures/bedroom.mp4` |
| Source SHA-256 | `1be76d5d19b066e8ad7c565d88a98e11a8f8d456a707508a7aa35390def70e30` |
| Prepared SHA-256 | `8007058483188405474f9ccb088008975ae5c131c71382de865e5300aa8cf1b4` |
| Prepared input | 3 seconds, 4 fps, 560 × 316 pixels, 12 frames |
| Prompt | `person` |
| Grounded SAM 2 bundle | `08c760b7eb21a0a435ad5821a392ffcda957b967e2156cf4dd886a830bfbde81` |
| SAM 3 bundle | `7470e803f820ca3af8f23fd2a098491d1ae9f53c1df40f2f62da0fb55ad79086` |

Both integrity manifests were verified file by file. Both archives also passed
safe-path, schema, mask-dimension, mask-reference, and RLE-expansion checks.

## Evidence result

| Measurement | Grounding DINO 1.0 + SAM 2.1 | SAM 3 Transformers |
| --- | ---: | ---: |
| Tracks | 2 | 2 |
| Observations | 24 | 24 |
| Frames with evidence | 12 / 12 | 12 / 12 |
| Track gaps | 0 | 0 |
| Mean consecutive-mask IoU | 0.3548 | 0.3541 |
| Mean allocated GPU memory | not measured | not measured |
| Peak allocated GPU memory | 4.059 GiB | 2.049 GiB |
| Peak reserved GPU memory | 9.715 GiB | 2.260 GiB |
| Peak process RSS | 3.690 GiB | 5.255 GiB |

After matching identities across the systems:

- adult mean cross-model mask IoU: **0.9634**;
- child mean cross-model mask IoU: **0.9080**;
- all matched observations mean mask IoU: **0.9357**;
- lowest matched IoU: **0.6368**, for the child at frame 8 during strong
  overlap/occlusion.

Visual inspection found no identity switch, duplicate track, missing mask, or
obvious semantic false positive. The adult masks are nearly interchangeable.
Most disagreement comes from the smaller, partially occluded child.

## Confidence semantics

The scores must not be compared as calibrated frame confidence.

Grounded SAM 2 repeats the two first-frame Grounding DINO seed scores
(0.8370 and 0.8076) across the propagated observations. SAM 3 reports 0.96484375
for both identities on every frame. These values describe different model
boundaries and neither archive currently provides an independently calibrated
per-frame visibility probability.

## Recorded timing

| Phase | Grounded SAM 2 | SAM 3 |
| --- | ---: | ---: |
| Model load | 7.125 s | 6.783 s |
| Grounding or prompt | 3.162 s | 0.004 s |
| Session initialization | 0.412 s | 0.245 s |
| Video propagation | 6.999 s | 13.000 s |
| Recorded model total | 17.697 s | 20.032 s |

The SAM 3 timing is **provisional and must not be used for a performance
conclusion**. Its archive records `torch.bfloat16` on a Tesla T4. NVIDIA
documents T4 Tensor Core acceleration for FP16/FP32, INT8, and INT4, not BF16.
The notebook incorrectly relied on PyTorch's software support check and has now
been corrected to select FP16 for compute capability 7.5.

Download timings are also excluded: the Grounded SAM 2 checkpoints were cached,
while the SAM 3 snapshot download took 25.7 seconds.

## Applied conclusion

On this clip, the two systems produce substantively the same evidence. There is
no accuracy result here that justifies choosing one over the other.

For Article 1:

- use **SAM 3 as the primary recorded demo path** because it exposes one
  text-prompted video model boundary and used substantially less T4 GPU memory;
- keep **Grounded SAM 2 as the ungated reproducible baseline**;
- make the article's real contribution the model-independent evidence graph,
  deterministic measurements, and inspectable citations;
- do not claim a SAM 3 speed advantage or general accuracy advantage from this
  run.

A later performance table may add one FP16 SAM 3 timing sample if a cached T4
session is available, but the present artifacts are sufficient for the article's
functional comparison and web demo.

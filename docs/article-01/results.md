# Article 1 — matched cup-crossing perception run

## Status

The controlled hero-scene run is complete. Grounding DINO 1.0 + SAM 2.1 and
SAM 3 Transformers processed byte-identical prepared input and produced valid,
complete evidence archives.

This is an applied smoke test on one short generated clip, not a general model
benchmark.

## Reproducibility

| Field | Value |
| --- | --- |
| Runtime | Google Colab, Tesla T4 (14.56 GiB), CUDA 12.8, Torch 2.11.0 |
| Source | `assets/article-01/sample.mp4` |
| Source provenance | Gemini-generated video supplied by the repository owner |
| Source SHA-256 | `4bc06ddbbcc711f3ec706f9d25cd7377638f3ef81b08c5d0142e6342db5f3060` |
| Prepared SHA-256 | `db1e6fb06413e31722e2678f8f6d3282cdeb50630a5ec94e18100f6366f68e49` |
| Prepared input | 10 seconds, 4 fps, 560 × 316 pixels, 40 frames |
| Prompt | `white cup` |
| Grounded SAM 2 archive | `109ee46c5ad7b41b3acd3842dd3c6b3d99f34aee011963b7113e913a59d6e327` |
| SAM 3 archive | `74b8d147f2546e50f8048a1e46ff5e78d411acaa211234086f7ea8b553bf20a3` |

Both archives passed safe-path, duplicate-entry, manifest-coverage, byte-length,
and SHA-256 validation. Their scene documents, mask dimensions, mask references,
and row-major RLE payloads were also validated before comparison.

## Evidence result

| Measurement | Grounding DINO 1.0 + SAM 2.1 | SAM 3 Transformers |
| --- | ---: | ---: |
| Tracks | 1 | 1 |
| Observations | 40 | 40 |
| Frames with evidence | 40 / 40 | 40 / 40 |
| Track gaps | 0 | 0 |
| Zone A interval | 0.00–3.75 s | 0.00–3.75 s |
| Zone A observed dwell | 4.00 s | 4.00 s |
| Zone B interval | 4.00–9.75 s | 4.00–9.75 s |
| Zone B observed dwell | 6.00 s | 6.00 s |
| Mean consecutive-mask IoU | 0.8595 | 0.8599 |

Both systems identify the same deterministic boundary: the cup is last observed
in A at 3.75 seconds and first observed in B at 4.00 seconds. The transition
resolution is one sampled frame, or 250 milliseconds.

## Cross-model agreement

The single cup tracks were matched observation-by-observation across all 40
frames.

| Measurement | Result |
| --- | ---: |
| Mean mask IoU | **0.9694** |
| Median mask IoU | 0.9692 |
| Minimum mask IoU | 0.9487 at frame 15 (3.75 s) |
| Maximum mask IoU | 0.9829 |
| Mean bounding-box IoU | 0.9648 |
| Mean mask-centroid distance | 0.447 px |
| Maximum mask-centroid distance | 1.068 px |
| Mean SAM 2 / SAM 3 mask-area ratio | 1.0176 |

The lowest mask overlap occurs while the cup is moving across the notebook near
the zone boundary. Even there, the systems agree on the object, identity, zone,
and supporting time range. Visual inspection found no identity switch, duplicate
track, missing mask, semantic false positive, or material mask leak.

## Confidence semantics

The scores must not be compared as calibrated frame confidence.

- Grounded SAM 2 repeats the first-frame Grounding DINO seed score across
  propagated observations.
- SAM 3 reports one repeated text-prompted track score across this recording.

The web explorer therefore labels them as a seed score and a track score rather
than presenting either as 40 independent confidence measurements.

## Recorded timing and memory

| Phase | Grounded SAM 2 | SAM 3 |
| --- | ---: | ---: |
| Checkpoint/model download | 14.325 s | 31.234 s |
| Model load | 15.719 s | 8.868 s |
| Grounding or prompt | 4.026 s | 0.001 s |
| Session initialization | 1.601 s | 0.511 s |
| Video propagation/inference | 15.958 s | 10.186 s |
| Recorded model total, excluding download | 37.303 s | 19.567 s |
| Peak allocated GPU memory | 3.283 GiB | 1.810 GiB |
| Peak reserved GPU memory | 6.242 GiB | 1.877 GiB |
| Peak process RSS | 3.876 GiB | 5.233 GiB |

These numbers describe two individual Colab runs with different dependency and
model boundaries. They are useful operational observations, not sufficient
evidence for a general speed or memory ranking.

## Setup observation

The Grounded SAM 2 run reported that its optional compiled `_C` extension could
not be imported, so SAM 2 skipped optional post-processing. Core propagation
completed normally. The warning is retained as setup evidence; the artifact
validation, quantitative comparison, and visual inspection found no material
failure in this scene.

## Applied conclusion

On the controlled cup-crossing scene, both perception stacks produce
substantively the same queryable evidence. The 0.9694 mean cross-model mask IoU
is supportive agreement, not ground-truth accuracy.

For Article 1:

- use **SAM 3 as the primary recorded interaction path**;
- keep **Grounded SAM 2 as the public, ungated reproducible baseline**;
- center the article on the model-independent evidence graph, deterministic
  measurements, and inspectable citations;
- present performance numbers as run provenance rather than a benchmark claim;
- use the browser explorer to make every claimed timestamp seekable and every
  perception overlay switchable.

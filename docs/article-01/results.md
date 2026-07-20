# Article 1 — provisional run results

## SAM 3 official-sample smoke test

Status: validated, pending a matched Grounded SAM 2 run.

- bundle: `article-01-hero-sam3-transformers.zip`
- bundle SHA-256: `2546fecb84578e16c6da3993be99cc3e2904a7db514bffe7befad6518672045b`
- runtime: Google Colab (reported by the runner)
- source: public `bedroom.mp4` from `hf-internal-testing/sam2-fixtures`
- prompt: `person`
- prepared input: 3 seconds, 4 fps, 560 × 316 pixels, 12 frames
- model: `facebook/sam3`
- integration: Transformers 5.14.1
- dtype: bfloat16
- model load: 51.389 s
- video inference: 13.181 s
- peak allocated GPU memory: 2.049 GiB
- result: two object identities, both present in all 12 frames
- evidence: 24 referenced masks; all present and structurally valid
- evidence gaps: none

Both identities received the identical score, 0.96484375, on every frame. Treat this
as a sequence/query-level score rather than evidence of calibrated frame-by-frame
confidence unless the upstream model documentation establishes otherwise.

The archive passed structural checks for safe paths, aligned frame columns,
normalized in-bounds boxes, mask dimensions, complete mask references, and RLE
run lengths summing to the full image area.

## Caveats

The original metrics did not record the GPU model, so this run must not yet be
labelled as a T4 benchmark. The notebook now records runtime, GPU name, total GPU
memory, Python, Torch, and Transformers versions for subsequent runs.

This is a smoke-test result, not a model comparison. The next accepted result must
run Grounding DINO 1.0 + SAM 2.1 on the identical prepared input and prompt.

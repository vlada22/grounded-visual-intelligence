# SAM 3 Transformers integration boundary

Article 1 uses the official `facebook/sam3` Transformers video path as its
primary recorded perception experience. Model access is gated through Hugging
Face, so Grounding DINO + SAM 2.1 remains the ungated reproducible baseline.
Neither path leaks PyTorch tensors or model lifecycle concerns into the Visual
Evidence Graph.

## Recorded-output contract

The Colab runner prepares the repository-owned source video deterministically,
loads `Sam3VideoModel` and `Sam3VideoProcessor`, adds the text prompt, and records
the streamed response for every propagated frame.

The portable `Sam3RecordedOutput` retains:

- the upstream persistent object ID;
- prompt/track probability metadata;
- pixel-space bounding boxes;
- frame indices and timestamps;
- one row-major binary-mask RLE artifact per visible observation;
- model revision, runtime, package, timing, memory, and source provenance.

## Repository boundary

`Sam3Adapter` converts the recording into the same model-independent `Scene`
document used by the Grounded SAM 2 adapter. It:

1. validates frame and column consistency;
2. groups observations by persistent object ID;
3. preserves the score without presenting it as calibrated frame confidence;
4. records model, version, task, prompt, and source provenance;
5. emits model-neutral tracks consumed by `SceneMemory`;
6. leaves masks as portable URI-addressed artifacts.

The core `gvi` package deliberately does not depend on Transformers, PyTorch,
CUDA, SAM 3, or Hugging Face. GPU execution belongs to the Colab boundary;
tests and the browser explorer operate on recorded JSON and RLE artifacts.

## Controlled Article 1 run

The matched run uses:

- `assets/article-01/sample.mp4` with a verified SHA-256;
- the `white cup` prompt;
- ten seconds at 4 fps and maximum width 560;
- FP16 on a Tesla T4;
- identical A/B zone definitions for both model paths.

The completed SAM 3 archive contains 40 visible observations with no track gap.
It agrees with the Grounded SAM 2 masks at 0.9694 mean IoU across all 40 matched
frames. See `results.md` for the controlled-run evidence and limitations.

## Deliberate omissions

Article 1 does not provide:

- arbitrary public uploads;
- live hosted GPU inference;
- browser-native model execution;
- checkpoint redistribution;
- a general SAM 2 versus SAM 3 benchmark.

The browser experience consumes only the validated recorded artifacts. This
keeps the publication reproducible while making the model boundary replaceable.

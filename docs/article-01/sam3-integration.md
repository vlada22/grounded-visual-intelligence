# SAM 3.1 integration boundary

SAM 3.1 is an optional GPU perception component. It must not leak PyTorch
tensors or model lifecycle concerns into the Visual Evidence Graph.

## Upstream contract

The official SAM 3.1 video example returns one response per propagated frame:

```text
response["frame_index"]
response["outputs"]["out_obj_ids"]
response["outputs"]["out_probs"]
response["outputs"]["out_boxes_xywh"]
response["outputs"]["out_binary_masks"]
```

`out_boxes_xywh` uses normalized coordinates. Object IDs persist across the
propagated masklet. The relevant upstream references are:

- https://github.com/facebookresearch/sam3/blob/main/examples/sam3.1_video_predictor_example.ipynb
- https://github.com/facebookresearch/sam3/blob/main/sam3/visualization_utils.py

## Repository contract

The inference worker materializes each binary mask as a separate row-major RLE
artifact and records its URI. It writes `Sam3RecordedOutput`, a JSON-safe
representation that retains the official object IDs, probabilities, and
normalized boxes.

The core adapter then:

1. validates frame and column consistency;
2. converts normalized XYWH boxes into pixel XYXY boxes;
3. groups observations by persistent SAM object ID;
4. marks observations below the configured confidence threshold as not visible;
5. removes tracks that never have a visible observation;
6. records model, version, task, and prompt provenance;
7. emits a model-independent `Scene` document.

## Deliberate omissions

`Sam3InferenceWorker` now owns:

- session creation and cleanup;
- text-prompt submission;
- streamed propagation;
- tensor-to-CPU conversion;
- binary-mask encoding;
- raw artifact manifests;

The worker accepts a predictor protocol, allowing its lifecycle and output
contract to be tested without a GPU. `build_official_predictor()` imports the
official model lazily when the upstream package and gated weights are available.

This checkpoint still does not install SAM 3.1, download gated weights, probe
video metadata, or prescribe GPU infrastructure. Real inference will add:

- video probing and frame validation;
- latency and peak-memory measurements;
- an environment-specific runner for the selected GPU;
- the first redistribution-safe prepared video.

This separation lets tests and the web application operate without a GPU while
keeping recorded model outputs reproducible.

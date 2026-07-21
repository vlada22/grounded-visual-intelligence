# Ask the Video, Not Just the VLM

## Building an evidence-grounded visual memory with SAM 3, Grounded SAM 2, deterministic tools, and frame-level citations

## Artifacts

- [GitHub repository](https://github.com/vlada22/grounded-visual-intelligence)
- [Live evidence explorer](https://vlada22.github.io/grounded-visual-intelligence/)

After years of working with video analytics, I keep returning to the same uncomfortable question: when a system gives a precise answer about a video, can it show where that answer came from?

A vision-language model can watch a video and give you a convincing description. The trouble starts when you ask it exactly when an object crossed a boundary, how long it occupied a zone, or which frames support the answer.

Description is a language task. Measurement is an evidence task.

That distinction led me to build [**Grounded Visual Intelligence**](https://github.com/vlada22/grounded-visual-intelligence): an experiment in turning video into structured, queryable evidence before asking a language model to explain anything. I wanted the system to follow one simple rule:

> Let perception models observe. Let deterministic tools measure. Let the LLM plan and explain—with citations.

This article turns that rule into a small working system. It uses [SAM 3](https://arxiv.org/abs/2511.16719) as the primary perception path and [Grounding DINO](https://arxiv.org/abs/2303.05499) plus [SAM 2](https://arxiv.org/abs/2408.00714) as a reproducible baseline. Both pipelines process the same ten-second video and emit the same model-independent evidence graph.

The contribution is not another segmentation model. It is a clean boundary between perception, measurement, and language.

**[MEDIA 1 — Upload `transition-evidence.mp4` here, then remove this marker]**

*The recorded SAM 3 track around the zone transition. The animation is generated from the validated notebook artifact, not from live browser inference.*

## A fluent answer is not yet a measured answer

Imagine asking a video assistant:

> When did the white cup move from the left side of the divider to the right side?

A multimodal model may answer “around four seconds.” That can be useful, but it leaves several engineering questions unanswered:

- What object identity was followed across frames?
- Which spatial rule defines “left” and “right”?
- What is the temporal resolution of the estimate?
- Can another component reproduce the answer without re-running the LLM?
- Can a reviewer seek directly to the supporting frames?

The problem is not limited to answers that are simply wrong. An answer may be approximately right and still be operationally weak because its evidence, semantics, and uncertainty are hidden.

For measurable video questions, I want a different contract. If the system can compute a timestamp from tracked observations, the language model should not have to estimate it from pixels.

**[MEDIA 2 — Upload `01-evidence-pipeline.png` here, then remove this marker]**

*Figure 1. The system boundary. GPU perception can be replaced without changing the evidence model or query tools.*

## The architecture: one portable evidence layer

The pipeline has four responsibilities.

### 1. Perception produces observations

The perception layer detects or prompts an object, segments it, and preserves its identity through the clip.

SAM 3 is a natural fit for the primary path because its Promptable Concept Segmentation task accepts concept prompts and returns masks with persistent identities across images and video. Its architecture combines image-level detection with a memory-based video tracker. The official Transformers video implementation exposes that path directly.

The baseline separates the same responsibilities. Grounding DINO converts a text prompt into an open-set detection on the first frame; SAM 2 then propagates the selected object through video using its streaming-memory architecture. This follows the broader [Grounded SAM](https://arxiv.org/abs/2401.14159) pattern of composing open-world models.

These are very different model stacks. The rest of the application should not have to care.

### 2. An evidence graph normalizes the output

Each adapter converts its native tensors and response objects into a portable scene document. The central records are deliberately small:

```text
Scene
  metadata: source, width, height, fps, duration
  tracks[]
    track_id
    concept
    observations[]
      frame_index
      timestamp_s
      bounding_box
      confidence
      mask_uri
  zones[]
  provenance[]
```

Masks are stored as row-major binary RLE artifacts. Model weights, CUDA tensors, and framework objects do not cross this boundary.

That design has a useful consequence: tests, analytical tools, and a browser explorer can work from recorded JSON and RLE files without importing PyTorch, Transformers, or either SAM implementation.

### 3. Deterministic tools answer measurable questions

The scene memory exposes operations such as:

- count instances of a concept;
- retrieve a track at a time range;
- measure observed occupancy in a zone;
- identify the last observation in one zone and the first in another;
- return the exact track, frames, and time range used as evidence.

So an answer is a value plus an evidence reference—not just a plausible sentence.

### 4. The LLM orchestrates and explains

The LLM can translate a user question into tool calls and turn the result into readable language. But it does not own the count, the transition rule, or the timestamp. Those remain deterministic and inspectable.

This is a modest kind of grounding, but it changes the reliability of the whole interaction: generation happens after measurement.

## The controlled scene

I chose a deliberately simple synthetic scene: a white cup moves from a region marked A to a region marked B, crossing a narrow divider while passing over a notebook. The scene is almost boring—and that is useful. A simple clip makes it easier to inspect every assumption before moving on to crowded, ambiguous video.

The source clip was generated with Gemini and then versioned in the repository. Both notebooks verified the same source SHA-256 and produced byte-identical prepared input:

- **Duration:** 10 seconds
- **Sampling:** 4 fps
- **Resolution:** 560 × 316
- **Frames:** 40
- **Prompt:** `white cup`
- **Prompt frame:** 0
- **Interactive correction:** none
- **GPU:** Tesla T4 on Google Colab

The A and B zones exclude a narrow normalized band around the divider. Zone membership uses the normalized center of the tracked bounding box. At 4 fps, the finest possible transition resolution in this run is 250 milliseconds.

This is an applied smoke test, not a general segmentation benchmark. I would rather make that boundary explicit than turn one clean example into a larger claim.

**[MEDIA 3 — Upload `02-transition-window.jpg` here, then remove this marker]**

*Figure 2. The evidence window. The cup is last observed in A at 3.75 s and first observed in B at 4.00 s.*

## The answer the system can defend

Both perception paths produced:

- one cup track;
- 40 observations over 40 frames;
- no missing evidence frames;
- no track gaps;
- the same A-to-B boundary.

The deterministic answer is:

> The white cup moved from zone A to zone B between **3.75 and 4.00 seconds**. It was last observed in A at frame 15 and first observed in B at frame 16. With 4 fps sampling, the transition is localized to a 250 ms window.

The measured occupancy was also identical:

- **Zone A:** both pipelines observed the cup from 0.00–3.75 s, for 4.00 s of sampled dwell.
- **Zone B:** both pipelines observed the cup from 4.00–9.75 s, for 6.00 s of sampled dwell.

“Observed dwell” matters here. It is the number of sampled observations assigned to the zone multiplied by the 250 ms sampling period. It should not be confused with a continuous, sub-frame physical measurement.

## How closely did the two model paths agree?

Finding the same cup was the expected part. The more interesting question was whether the two pipelines produced similar enough spatial evidence to support the same measurements. I matched their single cup tracks frame by frame and compared the binary masks, bounding boxes, and mask centroids.

Across all 40 frames:

- **Mean mask IoU:** 0.9694
- **Median mask IoU:** 0.9692
- **Minimum mask IoU:** 0.9487 at 3.75 s
- **Maximum mask IoU:** 0.9829
- **Mean bounding-box IoU:** 0.9648
- **Mean centroid distance:** 0.447 px
- **Maximum centroid distance:** 1.068 px

The lowest mask overlap occurs at the last zone-A observation, while the cup is moving over the notebook near the divider. Even there, both systems agree on the concept, identity, zone, and transition evidence.

**[MEDIA 4 — Upload `03-cross-model-agreement.png` here, then remove this marker]**

*Figure 3. Cross-model spatial agreement. The shaded region is the measured transition window.*

There is an important limit to this result: **agreement is not accuracy**.

The clip has no hand-annotated ground-truth masks. Two systems can agree and still be wrong in the same way. The comparison supports a narrower conclusion: for this controlled scene, either perception stack yields substantively the same downstream evidence and deterministic answer.

That is enough to validate the adapter boundary. It is not enough to declare a winner.

## Confidence is part of the schema—not a universal number

The two pipelines also expose why model metadata must retain its semantics.

In the Grounded SAM 2 recording, the confidence attached to propagated observations is the initial Grounding DINO seed score. In the SAM 3 recording, the repeated value is a text-prompted track score. Neither should be presented as 40 independently calibrated frame confidences, and the two values should not be compared directly.

The evidence explorer therefore labels them differently: **seed score** and **track score**.

This may sound like a fussy UI detail, but it is really a data-contract requirement. A generic field named `confidence` is only useful when its provenance and meaning travel with it.

## Operational observations, not a speed leaderboard

Both runs recorded phase timings and memory on a Colab T4.

Grounded SAM 2 took 37.30 seconds across model load, grounding, session initialization, and propagation. The corresponding SAM 3 path took 19.57 seconds. Peak allocated GPU memory was 3.28 GiB and 1.81 GiB respectively.

It is tempting to stop there and call SAM 3 faster and lighter. The data does not support that conclusion yet. The dependency boundaries differ, the model-loading paths differ, and these are single runs. Process RSS also moves in the opposite direction: 3.88 GiB for Grounded SAM 2 versus 5.23 GiB for SAM 3.

**[MEDIA 5 — Upload `04-run-profile.png` here, then remove this marker]**

*Figure 4. One-run operational provenance. Checkpoint download time is excluded from the phase chart.*

The Grounded SAM 2 notebook also reported that its optional compiled `_C` extension could not be imported, so optional post-processing was skipped. Core propagation completed normally, and archive validation plus visual inspection found no material failure in this scene. I kept the warning in the record. Reproducibility is more useful when it includes the awkward setup details too.

## Making the answer inspectable

The final interface is a static evidence explorer. It performs no hosted GPU inference and sends no request to an LLM. It loads the validated video, the two normalized scene tracks, and their 80 RLE masks.

The explorer lets a reader:

- seek to either cited boundary frame;
- scrub the evidence timeline;
- switch between SAM 3 and Grounded SAM 2;
- toggle masks, boxes, and zones;
- inspect coverage, gaps, scores, timing, and memory provenance.

SAM 3 is the default view because it provides the cleanest primary concept-prompted path. Grounded SAM 2 stays available as the ungated baseline.

The explorer is a dependency-free static application. You can [open the live evidence explorer](https://vlada22.github.io/grounded-visual-intelligence/) or inspect its implementation in the [GitHub repository](https://github.com/vlada22/grounded-visual-intelligence). The recorded GIF and figures in this article are reproducible from the two notebook archives.

## What this experiment proves—and what it does not

For me, the most useful result is architectural.

It shows that two different video-perception stacks can be reduced to one stable evidence contract. Once that contract exists, temporal and spatial questions become deterministic operations over tracks instead of open-ended guesses over pixels. The LLM becomes easier to trust because its factual burden is smaller.

This experiment does **not** yet prove that the system reduces VLM error in general. It does not evaluate a direct-VLM baseline, crowded scenes, multiple same-class instances, long occlusion, re-identification, camera motion, or real operational footage. It also lacks ground-truth masks and transition annotations.

Those are the next tests:

1. Build an annotated multi-scene evaluation set and compare direct VLM answers against tool-grounded answers.
2. Add multiple instances, occlusion, disappearance, and re-entry.
3. Separate identity uncertainty from mask uncertainty and temporal sampling uncertainty.
4. Extend the graph with depth, 3D geometry, and spatial relations.
5. Measure end-to-end answer quality, latency, and cost with and without the evidence layer.

## The larger idea

Video foundation models are becoming better at detecting, segmenting, and tracking concepts. That makes the layer above them more—not less—important.

Applications still need to preserve what was observed, when it was observed, how it was measured, and which artifact supports a claim. Otherwise even a powerful perception model becomes another opaque input to a text generator.

My preferred design principle is now straightforward:

> If a visual fact can be measured, do not ask the language model to invent it. Give the model a tool, preserve the evidence, and make the answer seekable.

That is the difference between a video system that sounds correct and one that can show its work.

I’ve published the code, notebooks, evidence artifacts, and [interactive explorer](https://vlada22.github.io/grounded-visual-intelligence/) in the [GitHub repository](https://github.com/vlada22/grounded-visual-intelligence). Where would you draw the boundary between perception, measurement, and language reasoning in a production visual system?

---

### Reproducibility note

The complete [Grounded Visual Intelligence repository](https://github.com/vlada22/grounded-visual-intelligence) contains the model-independent Python package, adapters, tests, two Colab notebooks, the controlled source clip, validated result documentation, the static explorer, and the plotting source used for the figures above. Exact source, prepared-input, checkpoint, archive, and model-revision digests are recorded in the run artifacts and project results document.

### References

- Carion et al., [SAM 3: Segment Anything with Concepts](https://arxiv.org/abs/2511.16719), 2025/2026.
- Ravi et al., [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714), 2024.
- Liu et al., [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499), 2023/2024.
- Ren et al., [Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks](https://arxiv.org/abs/2401.14159), 2024.
- Hugging Face Transformers, [SAM3 Video documentation](https://huggingface.co/docs/transformers/model_doc/sam3_video).


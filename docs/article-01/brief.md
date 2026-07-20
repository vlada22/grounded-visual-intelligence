# Article 1 brief

## Working title

**Ask the Video, Not Just the VLM: Building an Evidence-Grounded Visual Memory**

## Reader promise

Show how open-vocabulary video perception can become a queryable evidence
system: the vision model observes, deterministic tools measure, and an LLM
plans and explains without fabricating counts or timestamps.

## Central question

Can a language model answer temporal questions about a video more reliably
when it operates over structured visual evidence rather than sampled frames
alone?

## Demonstrated pipeline

1. A prepared video and an open-vocabulary concept prompt enter SAM 3.1.
2. Masks, boxes, confidence values, and persistent object IDs are normalized
   into the Visual Evidence Graph.
3. Deterministic tools compute counts, visibility intervals, zone visits,
   ordering, and supporting evidence.
4. An LLM selects tools and writes an answer.
5. Every factual answer links to timestamps and highlighted frames.

## Scope

### Included

- two or three prepared 10-30 second videos;
- SAM 3.1 offline inference;
- a versioned evidence schema;
- six to ten deterministic tools;
- an LLM tool-calling adapter;
- approximately 25 manually verified questions;
- track, confidence, occupancy, and evaluation plots;
- a static web application backed by precomputed vision artifacts;
- a short publication-quality hero video.

### Explicitly deferred

- 3D reconstruction and metric distance;
- identity recovery after long disappearances;
- model training or fine-tuning;
- arbitrary public uploads;
- live GPU inference;
- browser-native model execution;
- a generic agent framework or vector database.

## Claims we may test

1. Tool-grounded answers reduce numeric and temporal errors for questions that
   can be expressed as deterministic operations over tracks.
2. Evidence links make failures inspectable even when the final answer is
   incorrect.
3. Direct VLM reasoning remains useful for semantic attributes that are absent
   from the evidence graph, but it should not be the source of measured facts.

These are hypotheses until the evaluation is complete. The article conclusion
must follow the results.

## Hero scene

A controlled tabletop sequence with two toy vehicles, cups or boxes that cause
short occlusions, one marked zone, one entry or exit, and modest camera motion.
The final scene must be safe to redistribute and visually legible without text.

## Completion criteria

- reproducible inference for all prepared videos;
- validated scene artifacts checked into a release bundle;
- deterministic results for the supported question taxonomy;
- direct-VLM and grounded-system results recorded from the same questions;
- a polished browser demo in which cited timestamps seek the video;
- a Medium article, LinkedIn post, repository release, and hero animation.


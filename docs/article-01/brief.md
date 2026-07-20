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

1. SAM 3 tracks a text-prompted concept through the prepared hero video.
2. Grounding DINO plus SAM 2.1 runs as the ungated comparison baseline.
3. Boxes, masks, confidence metadata, and track IDs from either system are
   normalized into the same Visual Evidence Graph.
4. Deterministic tools compute counts, visibility intervals, zone visits,
   ordering, and supporting evidence.
5. An LLM selects tools and writes an answer.
6. Every factual answer links to timestamps and highlighted frames.

## Scope

### Included

- two or three prepared 10-30 second videos;
- Grounding DINO 1.0 plus SAM 2.1 offline inference;
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

A ten-second Gemini-generated tabletop sequence in which a white cup moves from
marked zone A to zone B across a visible blue divider, with hand and notebook
occlusion. A red apple appears late as a new-object discovery probe. The source
is supplied and approved for publication by the repository owner.

## Completion criteria

- reproducible inference for all prepared videos;
- validated scene artifacts checked into a release bundle;
- deterministic results for the supported question taxonomy;
- direct-VLM and grounded-system results recorded from the same questions;
- a polished browser demo in which cited timestamps seek the video;
- a Medium article, LinkedIn post, repository release, and hero animation.

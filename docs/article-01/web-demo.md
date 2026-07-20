# Interactive evidence explorer

The Article 1 web experience is a static, fixture-backed demonstration of the
grounding contract. It deliberately performs no GPU inference and makes no
network request to a language model. That separation keeps the interaction
fast, reproducible, and suitable for embedding beside the article.

[Open the current private checkpoint](https://grounded-visual-intelligence-lab.vlatko-nikol-0153.chatgpt.site)

## Demonstrated interaction

- submit the representative temporal question;
- inspect the deterministic `1.0–1.5 s` answer;
- seek to cited boundary frames;
- highlight the relevant track or zone;
- toggle perception overlays;
- scrub the timeline by pointer or keyboard;
- compare the answer with the occupied interval.

The editorial layout treats the video as the primary artifact and the query,
answer, evidence chips, and occupancy interval as a compact provenance chain.
The generated tabletop plate is illustrative until the first prepared video and
recorded SAM 3.1 output are available.

## Next data binding

The Colab notebook downloads `sam3-output.json`, mask artifacts, and run metrics.
The Python adapter converts that recorded output into a model-independent
`Scene`. A small web export step will then replace the illustrative constants
with scene metadata, tracks, evidence references, and pre-rendered keyframes.

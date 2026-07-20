# Interactive evidence explorer

The Article 1 web experience is a static, fixture-backed demonstration of the
grounding contract. It deliberately performs no GPU inference and makes no
network request to a language model. That separation keeps the interaction
fast, reproducible, and suitable for embedding beside the article.

[Open the current private checkpoint](https://grounded-visual-intelligence-lab.vlatko-nikol-0153.chatgpt.site)

## Demonstrated interaction

- ask when the white cup moves from zone A to zone B;
- inspect the deterministic transition time and dwell result;
- seek to cited boundary frames;
- highlight the relevant track or zone;
- toggle perception overlays;
- scrub the timeline by pointer or keyboard;
- compare the answer with the occupied interval.

The editorial layout treats the video as the primary artifact and the query,
answer, evidence chips, and occupancy interval as a compact provenance chain.
The publishable hero source is the repository owner's Gemini-generated
ten-second cup-crossing video. SAM 3 is the primary recorded path and Grounded
SAM 2 remains the ungated baseline.

## Next data binding

The two Colab notebooks export the prepared clip, `scene.json`, masks,
per-frame measurements, deterministic zone visits, preview media, and integrity
metadata. The next web export binds those recorded artifacts to the GitHub-hosted
explorer and replaces all illustrative constants.

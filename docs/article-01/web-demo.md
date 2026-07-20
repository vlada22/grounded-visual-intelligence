# Interactive evidence explorer

The Article 1 web experience is a static, artifact-backed demonstration of the
grounding contract. It performs no GPU inference and makes no network request to
a language model. That separation keeps the interaction fast, reproducible, and
suitable for linking from Medium and LinkedIn.

[Open the GitHub Pages evidence explorer](https://vlada22.github.io/grounded-visual-intelligence/)

## Repository source

- application: `web/index.html`, `web/styles.css`, and `web/app.js`;
- recorded evidence: `web/data/evidence.json`;
- publishable source video: `assets/article-01/sample.mp4`;
- deployment: `.github/workflows/pages.yml` through GitHub Pages.

The deployment workflow assembles the static site and copies the repository
video into the Pages artifact. The evidence payload contains the two normalized
scene tracks and their portable RLE masks, so the browser renders model-specific
overlays without storing a second video or requiring a backend.

## Demonstrated interaction

- ask when the white cup moves from zone A to zone B;
- inspect the deterministic transition window and observed dwell result;
- seek to cited boundary frames;
- switch between SAM 3 and Grounded SAM 2 evidence;
- toggle masks, boxes, and A/B zones independently;
- scrub the timeline by pointer or keyboard;
- compare coverage, gaps, runtime provenance, memory, and cross-model agreement.

The editorial layout treats the video as the primary artifact and the question,
answer, evidence chips, occupancy interval, and matched-run summary as one compact
provenance chain.

## Data binding

The illustrative constants from the first design checkpoint have been replaced
with the validated cup-crossing recordings. The browser payload is derived from:

- the byte-identical prepared-input digest;
- both model-independent `scene.json` documents;
- 80 portable frame-mask RLE records;
- both run-metrics and analysis summaries;
- the independently computed cross-model comparison.

SAM 3 is the default view. Grounded SAM 2 remains available as the ungated
baseline, and the interface deliberately distinguishes their incompatible score
semantics.

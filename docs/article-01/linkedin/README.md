# LinkedIn publishing kit — Article 01

This folder contains the LinkedIn-ready version of **Ask the Video, Not Just the
VLM**. The canonical research article remains in `docs/article-01/article.md`.

## Article metadata

- **Title:** Ask the Video, Not Just the VLM
- **Subtitle:** Building an evidence-grounded visual memory with SAM 3, Grounded
  SAM 2, deterministic tools, and frame-level citations
- **Suggested article URL:** `evidence-grounded-video-memory`
- **SEO title:** Evidence-Grounded Video Memory with SAM 3 and SAM 2
- **SEO description:** A practical architecture for turning video into queryable
  evidence, deterministic measurements, and frame-level citations before an LLM
  explains the result.

## Files

- `linkedin-article.md` — article text with LinkedIn-safe lists instead of tables
- `feed-post.md` — short post to publish with the article
- `assets/linkedin-cover.png` — 1200 × 644 cover image
- `assets/transition-evidence.mp4` — short transition animation for Media 1
- [`../assets/01-evidence-pipeline.png`](../assets/01-evidence-pipeline.png) — Media 2
- [`../assets/02-transition-window.jpg`](../assets/02-transition-window.jpg) — Media 3
- [`../assets/03-cross-model-agreement.png`](../assets/03-cross-model-agreement.png) — Media 4
- [`../assets/04-run-profile.png`](../assets/04-run-profile.png) — Media 5

## Publishing sequence

1. On LinkedIn desktop, choose **Write article**.
2. Upload `assets/linkedin-cover.png` as the cover image.
3. Enter the title above. Use the subtitle as the first subheading.
4. Paste `linkedin-article.md` into the editor. Apply heading styles to the lines
   beginning with `##` and `###`; LinkedIn will not interpret Markdown syntax
   automatically in every paste path.
5. Replace each numbered `MEDIA` marker with the matching file below. Delete the
   marker after uploading the media.
6. Format the `Scene` schema block with LinkedIn's **Code** style.
7. Confirm that both artifact links near the top are clickable.
8. Under **Manage → Settings**, set the suggested article URL, SEO title, and SEO
   description.
9. Preview the article. Search for `MEDIA` and `##` to make sure no publishing
   markers or Markdown heading characters remain.
10. Publish, then use `feed-post.md` as the accompanying network post.

## Media order, alt text, and captions

### Cover

- **File:** `assets/linkedin-cover.png`
- **Alt text:** Ask the Video, Not Just the VLM — an evidence-grounded visual
  memory pipeline that observes, measures, and cites video evidence.

### Media 1 — transition animation

- **File:** `assets/transition-evidence.mp4`
- **Alt text:** A tracked white cup moves from zone A to zone B while SAM 3 masks,
  a bounding box, and the zone divider are overlaid.
- **Caption:** The recorded SAM 3 track around the zone transition. Generated from
  the validated notebook artifact, not from live browser inference.

### Media 2 — architecture

- **File:** `../assets/01-evidence-pipeline.png`
- **Alt text:** Video enters a replaceable perception layer, becomes an evidence
  graph, and is measured by deterministic tools before an LLM explains the result.
- **Caption:** Figure 1. GPU perception can be replaced without changing the
  evidence model or query tools.

### Media 3 — transition frames

- **File:** `../assets/02-transition-window.jpg`
- **Alt text:** Four sampled frames show the white cup approaching the divider,
  last appearing in A at 3.75 seconds, first appearing in B at 4.00 seconds, and
  becoming established in B.
- **Caption:** Figure 2. The measured transition is localized to one sampled frame,
  or 250 milliseconds.

### Media 4 — agreement plot

- **File:** `../assets/03-cross-model-agreement.png`
- **Alt text:** Line chart of mask IoU, bounding-box IoU, and mask-centroid distance
  across 40 matched observations from Grounded SAM 2 and SAM 3.
- **Caption:** Figure 3. Cross-model spatial agreement; the shaded region is the
  measured transition window.

### Media 5 — runtime provenance

- **File:** `../assets/04-run-profile.png`
- **Alt text:** Two charts compare recorded model-path phase times and peak memory
  for single Grounded SAM 2 and SAM 3 runs on a Colab Tesla T4.
- **Caption:** Figure 4. One-run operational provenance, not a general performance
  benchmark.

## Final checks

- The live demo points to <https://vlada22.github.io/grounded-visual-intelligence/>.
- The repository points to
  <https://github.com/vlada22/grounded-visual-intelligence>.
- The article says cross-model **agreement**, not ground-truth accuracy.
- The Grounded SAM 2 seed score and SAM 3 track score are not compared as
  calibrated frame confidence.
- The runtime chart is described as one-run provenance, not a leaderboard.
- All five media markers have been removed.
- The article preview works on desktop and mobile widths.


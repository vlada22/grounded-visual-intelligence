# Article 1 publishable hero video

## Source

- filename: `sample.mp4`
- provenance: Gemini-generated video supplied by the repository owner
- publication status: approved by the repository owner for the Article 1 project
- SHA-256: `4bc06ddbbcc711f3ec706f9d25cd7377638f3ef81b08c5d0142e6342db5f3060`
- duration: 10.005 seconds
- video: H.264, 1280 × 720, 24 fps, 240 frames
- audio: AAC; intentionally removed by the comparison preprocessing step

Keep the original source outside Git until the final site asset layout is chosen.
The notebooks reject any upload whose source digest does not match the approved
file.

## Prepared comparison

Both perception runners apply the same deterministic transform:

- first 10 seconds;
- 4 fps;
- maximum width 560 pixels;
- aspect ratio preserved with even dimensions;
- H.264, YUV 4:2:0;
- no audio.

The local reference transform produces a 560 × 316, 40-frame clip. Its encoded
digest can vary with the FFmpeg build, so the two returned archives must still
be checked for byte-identical prepared inputs before comparing models.

## Scene semantics

The tabletop contains a visible vertical blue divider:

- zone A: normalized x range 0.00–0.49;
- neutral boundary band: 0.49–0.51;
- zone B: normalized x range 0.51–1.00.

The controlled prompt is `white cup`. The cup begins in A, crosses the divider
while interacting with the notebook and hand, and ends in B. This supports
first/last-seen, zone-entry, ordering, dwell, transition, occlusion, and evidence
citation examples.

A red apple appears late and is absent from frame zero. It is not included in
the controlled cup prompt. It provides a useful follow-up probe for the
first-frame-only discovery limitation of the Grounding DINO + SAM 2 pipeline.

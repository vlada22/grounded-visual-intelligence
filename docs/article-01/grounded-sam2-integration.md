# Grounding DINO + SAM 2.1 integration

Article 1 uses an ungated composition rather than relying on discretionary
checkpoint approval:

1. Grounding DINO 1.0 detects every first-frame instance matching the text
   prompt and provides an initialization box plus grounding score.
2. Each box receives a stable local object ID and initializes SAM 2.1's video
   predictor.
3. SAM 2.1 propagates a mask for every initialized object.
4. The runner derives pixel XYXY boxes and mask areas, encodes each non-empty
   mask as portable row-major RLE, and writes `GroundedSam2RecordedOutput`.
5. `GroundedSam2Adapter` converts the recording into the model-independent
   Visual Evidence Graph.

Detector and tracker provenance remain separate. The grounding score describes
the initial text-to-object match; it is not misrepresented as a per-frame
tracking probability. Missing or empty propagated masks become gaps in a track
rather than invented observations.

## Public checkpoints

- SAM 2.1 checkpoints are downloaded from Meta's public
  `dl.fbaipublicfiles.com/segment_anything_2` release path.
- Grounding DINO Swin-T is downloaded from the official Grounding DINO GitHub
  release.

The Colab notebook uses only those public URLs. A read-only GitHub token is
temporarily needed to clone this private article repository, not to obtain model
weights.

## Known limitation

The compact runner grounds objects only on the selected prompt frame. Objects
that first enter later cannot be discovered until we add periodic re-grounding
and track-ID reconciliation. This is an explicit experimental variable for the
occlusion and re-identification follow-up article.

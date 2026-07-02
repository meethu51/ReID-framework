# Limitations

This is an educational demo, not a production system.

## Intentionally simplified

- The detector is a toy color/blob detector.
- The embedding module is a deterministic stub.
- There is no trained ReID neural network by default.
- There is no camera calibration or world-plane geometry.
- There is no private dataset or model weight.
- There are no deployment integrations.

## Known technical limits

- Similar-looking people may still be confused.
- Bad bounding boxes reduce embedding quality.
- Manual zones require sensible placement.
- Webcams and videos may run at different frame rates.
- The global ID manager is intentionally conservative and simple.

## Extension ideas

- Plug in a real detector.
- Replace the embedding stub with a real ReID model.
- Add camera calibration.
- Add async frame ingest.
- Add structured JSONL diagnostics.
- Add unit tests for custom production extensions.


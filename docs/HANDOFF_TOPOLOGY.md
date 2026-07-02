# Handoff Topology

Handoff topology means the system uses camera layout and manually drawn zones to decide whether a cross-camera identity match is physically plausible.

## Manual zones

A zone is a rectangle in image coordinates:

```json
{
  "camera_id": "cam1",
  "zone_id": "cam1_exit",
  "type": "rect",
  "xyxy": [480, 80, 630, 420]
}
```

The demo computes:

- whether the track center is inside the zone,
- how much the box overlaps the zone,
- a simple confidence score.

## Why zones help

Appearance-only matching can over-merge identities. Handoff zones reduce false links by asking:

- Did a person appear near a plausible camera boundary?
- Is the source/destination camera pair allowed?
- Is the timing plausible?
- Is the embedding similarity compatible?

## Editor

```bash
python src/zone_editor.py --source sample.mp4 --camera-id cam1 --output examples/cam1_zone.json
```

Controls:

| Key | Action |
|---|---|
| Mouse drag | Draw rectangle |
| `r` | Reset |
| `s` | Save |
| `q` | Quit |


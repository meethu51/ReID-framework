# System Overview

ReID-Lite is a compact public demo of multi-camera person identity handoff. It is designed for education, portfolio review, and architecture discussion.

The system keeps three concepts separate:

1. Detection finds person-like regions in each frame.
2. Local tracking maintains short-term IDs inside one camera.
3. Global ReID links local tracks across cameras only when evidence is strong enough.

## Public-safe scope

This repo does not include production code, private datasets, real camera information, proprietary models, or deployment configuration. All examples use generic terms such as camera, person, operator, and zone.

## Core pipeline

```text
camera/video/synthetic input
  -> simple detector
  -> local tracker
  -> embedding stub
  -> topology handoff scoring
  -> global ID manager
  -> visualization/diagnostics
```

## Identity layers

- Local ID: valid only inside one camera.
- Global ID: public demo identity shared across cameras.
- Handoff candidate: temporary evidence that two local tracks may be the same person.
- Reacquired/remapped ID: demo concept for correcting local fragmentation.


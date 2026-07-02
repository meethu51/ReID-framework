# Sanitization Report

This folder was created as a public-safe portfolio version of the ReID project.

## Scope

Repository folder:

```text
public_reid_lite/
```

This repository is intentionally independent from the internal project. The source files were written as simplified public demo code rather than copied production modules.

## Removed or omitted

The public repo intentionally omits:

- proprietary detector implementation;
- private video footage;
- private datasets and crops;
- model weights and cached model files;
- internal deployment configuration;
- production RTSP URLs;
- credentials, keys, environment files, and tokens;
- private logs and output videos;
- operational site details;
- private integrations and review tooling;
- internal architecture documents.

## Replaced with public-safe equivalents

| Internal/private concept | Public-safe equivalent |
|---|---|
| Production detector | `simple_detector.py` toy blob detector |
| Real ReID model | `reid_embedding_stub.py` deterministic embedding stub |
| Real cameras | Webcam, local video, or synthetic generated frames |
| Deployment handoff zones | Example JSON in `examples/sample_handoff_zones.json` |
| Production global identity logic | Simplified `global_id_manager.py` |
| Internal terminology | Generic terms: camera, person, operator, zone |

## Sensitive-string scan summary

The public folder was scanned for:

- organization names;
- internal paths;
- IP-address-like strings;
- credential-like strings;
- AWS credential markers;
- real RTSP credential patterns;
- private deployment terminology;
- internal production module names.

Result:

- No internal filesystem paths were found.
- No private production module imports were found.
- No model weights, videos, logs, crops, caches, or datasets are included.
- The only RTSP string present is the intentional placeholder:

```text
rtsp://username:password@camera-ip:554/stream
```

This placeholder is included in documentation to show users where their own local test source would go. It is not a real credential or camera URL.

## Large-file check

The repo contains only Markdown, Python source, JSON examples, license/security text, `.gitignore`, and `requirements.txt`. No large binary files are included.

## Independence check

The public demo runs from inside `public_reid_lite/` and imports only its own `src/` modules plus public dependencies listed in `requirements.txt`.

Smoke test used:

```bash
python src/demo_multi_camera_reid.py --synthetic --camera-ids cam1 cam2 --zone-file examples/sample_handoff_zones.json --max-frames 20
```

## Disclaimer

This is a public-safe educational/demo version and not the production system.

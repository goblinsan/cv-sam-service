# Copilot Instructions

This repository is a GPU-backed image service intended to be deployed by `gateway-control-plane` as a `container-service` using `build.strategy: repo-compose`.

## Deployment contract

- Keep `docker-compose.yml`, `Dockerfile`, and `README.md` aligned.
- The service must continue listening on container port `5201`.
- The published host port must remain configurable through `HOST_PORT`.
- Health checks must continue to use `GET /api/health`.
- `GET /api/health` is allowed to return HTTP 200 while `ready=false` during model warmup.
- The service must remain compatible with NVIDIA GPU execution.

## Configuration rules

- Do not hardcode private LAN IPs, hostnames, node IDs, usernames, or local volume paths.
- Use placeholders in docs and examples for local operator values.
- If you add a new environment variable, update:
  - `docker-compose.yml`
  - `README.md`
  - any workload manifest example under `docs/`

## API stability

- Keep `/api/segment`, `/api/segment/auto`, `/api/analyze`, `/api/extract-palette`, and `/api/transform` stable unless the change is intentional and documented.
- Preserve both upload-based and `image_url`-based image inputs.
- Treat URL fetching as security-sensitive. Maintain SSRF protections and size limits.

## Agent-facing docs

- Keep `docs/workflow-action-contract.md` and `docs/agent-tool-openapi.yml` in sync with the implemented API.
- Do not claim control-plane or gateway-api changes are complete unless they are implemented in those separate repositories.

## Validation

Before considering a task complete, prefer verifying:

- Python syntax for app and tests
- Compose shape remains valid
- README examples still match the running API

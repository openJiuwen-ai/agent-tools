# Marketplace Tools Docker

This image is built for the `marketplace` module using a multi-stage wheel build.

## Build

Run from repository root:

```powershell
cd D:\Workspace\agent-tools

docker build -f docker/Dockerfile.marketplace-tools -t marketplace-tools:0.1.1 `
  marketplace
```

Notes:
- Build context is `marketplace`, so `marketplace/.dockerignore` is applied.
- Dockerfile automatically installs the wheel generated in `/app/dist` (no `VERSION` build-arg needed).
- Build does not depend on `uv.lock` (uses `uv sync` instead of `uv sync --frozen`).

## Run

```powershell
cd D:\Workspace\agent-tools
mkdir marketplace\data -Force

docker run --rm -p 8100:8100 `
  -v "${PWD}\marketplace\data:/app/data" `
  marketplace-tools:0.1.1
```

Endpoints:
- Docs: `http://localhost:8100/api/docs`
- Health: `http://localhost:8100/api/health`

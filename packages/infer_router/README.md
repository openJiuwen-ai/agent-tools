# openjiuwentools-infer-router

KV Cache aware inference routing system for vLLM and SGLang.

## Features

- **Dual Engine Support**: Compatible with both vLLM (PagedAttention) and SGLang (RadixAttention)
- **Real-time Cache Awareness**: Real-time synchronization of KV Cache status across instances
- **Intelligent Prefix Matching**: Select optimal instance based on prompt content
- **P/D Separation Support**: Support cache transfer in Prefill/Decode separation architecture
- **Non-invasive Integration**: Interact with inference engines via Sidecar mode

## Installation

```bash
pip install openjiuwentools-infer-router
```

Or install with etcd support:

```bash
pip install openjiuwentools-infer-router[etcd]
```

## Quick Start

```python
from openjiuwentools.infer_router.api.server import app
import uvicorn

uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Configuration

Set environment variables or create `.env` file:

```bash
WORKER_DISCOVERY_TYPE=config
WORKER_CONFIG_PATH=workers.yaml
```

## Documentation

See [documentation](https://gitcode.com/openJiuwen/agent-tools) for more details.

## License

Apache-2.0

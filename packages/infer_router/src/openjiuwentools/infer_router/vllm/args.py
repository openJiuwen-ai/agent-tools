import argparse


def _add_vllm_args(parser: argparse.ArgumentParser, *, model_required: bool = True) -> None:
    group = parser.add_argument_group("vLLM engine")
    group.add_argument("--model", type=str, required=model_required, help="Model name or path")
    group.add_argument("--tensor-parallel-size", type=int, default=1)
    group.add_argument("--pipeline-parallel-size", type=int, default=1)
    group.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    group.add_argument("--max-model-len", type=int, default=None)
    group.add_argument("--max-num-seqs", type=int, default=256)
    group.add_argument("--dtype", type=str, default="auto", choices=["auto", "half", "float16", "bfloat16", "float32"])
    group.add_argument("--quantization", type=str, default=None)
    group.add_argument("--trust-remote-code", action="store_true", default=False)
    group.add_argument("--enable-prefix-caching", action="store_true", default=False)
    group.add_argument(
        "--prefix-caching-hash-algo",
        type=str,
        default="sha256",
        choices=["builtin", "sha256", "sha256_cbor"],
        help="Prefix cache hash algorithm; use sha256 for deterministic cross-worker KVBM hashes",
    )
    group.add_argument(
        "--prefix-caching-hash-seed",
        type=int,
        default=0,
        help="Shared PYTHONHASHSEED used by vLLM to initialize the prefix-cache hash chain",
    )
    group.add_argument("--disable-log-requests", action="store_true", default=False)
    group.add_argument("--kv-events-config", type=str, default=None, help="JSON config for KV cache event reporting")
    group.add_argument("--kv-transfer-config", type=str, default=None, help="JSON config for KV cache transfer")


def _add_jiuwen_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Jiuwen worker")
    group.add_argument("--host", type=str, default="127.0.0.1", help="Server host; set explicitly for remote workers")
    group.add_argument("--api-key", default=None, help="Worker bearer key; prefer JIUWEN_WORKER_API_KEY environment")
    group.add_argument("--max-request-body-bytes", type=int, default=64 * 1024 * 1024)
    group.add_argument("--port", type=int, default=8001, help="Server port")
    group.add_argument(
        "--served-model-name", type=str, default=None,
        help="Alias exposed via API (defaults to --model)",
    )
    group.add_argument("--uvicorn-log-level", type=str, default="info")
    group.add_argument("--request-plane", type=str, default="http", choices=["http", "tcp"],
                        help="Request/response transport: http (FastAPI) or tcp (length-prefixed JSON)")
    group.add_argument("--etcd-endpoints", type=str, default=None,
                        help="etcd server address for service registration (e.g. localhost:2379)")
    group.add_argument("--service-name", type=str, default=None,
                        help="Service name registered in etcd (defaults to --served-model-name or --model)")
    group.add_argument("--registry-ttl", type=int, default=10,
                        help="etcd lease TTL in seconds (default: 10)")
    group.add_argument("--kv-relay-endpoint", type=str, default=None,
                        help="ZMQ PUB endpoint to relay KV cache events to router (e.g. tcp://*:5560)")
    group.add_argument("--worker-mode", type=str, default="aggregated",
                        choices=["aggregated", "prefill", "decode"],
                        help="Worker mode: aggregated (full inference), prefill only, or decode only")
    group.add_argument("--metrics-interval", type=float, default=1.0,
                        help="Interval in seconds for publishing worker metrics (default: 1.0)")
    group.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO)")
    group.add_argument("--tool-call-parser", type=str, default=None,
                        help="Tool call parser name, e.g. hermes (Qwen3), llama3_json, mistral")
    group.add_argument("--reasoning-parser", type=str, default=None,
                        help="Reasoning parser name, e.g. qwen3, deepseek_r1, granite")
    group.add_argument(
        "--enable-auto-tools",
        "--enable-auto-tool-choice",
        dest="enable_auto_tools",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable automatic tool calling; defaults to enabled when a parser is selected",
    )
    group.add_argument(
        "--auto-detect-parsers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Infer tool/reasoning parsers for known model families",
    )
    group.add_argument("--kvbm-worker-enabled", action="store_true", default=False,
                        help="Enable worker-side KVBM runtime when a KV provider is available")
    group.add_argument("--kvbm-worker-disk-root", type=str, default="/tmp/jiuwen-kvbm-worker",
                        help="Directory for worker-side KVBM disk offload")
    group.add_argument("--kvbm-clean-disk-on-start", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Clean old worker-side KVBM disk payloads at startup")
    group.add_argument("--kvbm-npu-high-watermark-blocks", type=int, default=0,
                        help="NPU cached block high watermark for KVBM offload")
    group.add_argument("--kvbm-ram-high-watermark-bytes", type=int, default=0,
                        help="RAM byte high watermark for KVBM disk spill")
    group.add_argument("--kvbm-worker-interval", type=float, default=5.0,
                        help="Interval in seconds for worker-side KVBM watermark checks")
    group.add_argument("--kvbm-block-size", type=int, default=16,
                        help="Fallback KV block size before KVBM infers it from real tensors")
    group.add_argument("--kvbm-worker-provider", type=str, default="auto",
                        choices=["auto", "vllm-ascend-0.18", "vllm-ascend-0.23"],
                        help="Worker-side KVBM provider; auto detects installed engine versions")
    group.add_argument("--kvbm-worker-diagnostics", action="store_true", default=False,
                        help="Print read-only KVBM provider diagnostics during worker startup")
    group.add_argument("--kvbm-worker-locator-depth", type=int, default=16,
                        help="Max object-tree depth for locating vLLM-Ascend KV caches")
    group.add_argument("--kvbm-auto-register-physical-blocks", action="store_true", default=False,
                        help="Register all physical KV blocks in the engine-core KVBM dataplane")
    group.add_argument("--kvbm-auto-enforce-watermarks-on-init", action="store_true", default=False,
                        help="Run one KVBM watermark pass immediately after KV cache initialization")
    group.add_argument("--kvbm-prefetch-host", type=str, default="127.0.0.1",
                        help="Host for the engine-core KVBM prefetch control server")
    group.add_argument("--kvbm-prefetch-port", type=int, default=None,
                        help="Port for the engine-core KVBM prefetch control server; default is worker port + 10000")
    group.add_argument("--kvbm-benchmark-enabled", action="store_true", default=False,
                        help="Enable isolated HIXL versus TCP staging transfer benchmark")
    group.add_argument("--kvbm-benchmark-tcp-host", type=str, default="127.0.0.1",
                        help="Bind host for the benchmark-only TCP staging source server")
    group.add_argument("--kvbm-benchmark-tcp-port", type=int, default=0,
                        help="Bind port for the benchmark-only TCP staging source server")
    group.add_argument("--kvbm-benchmark-max-blocks", type=int, default=64,
                        help="Maximum blocks accepted by one transfer benchmark request")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jiuwen vLLM Worker")
    _add_vllm_args(parser)
    _add_jiuwen_args(parser)

    return parser.parse_args()

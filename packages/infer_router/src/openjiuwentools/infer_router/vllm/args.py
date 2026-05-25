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
    group.add_argument("--disable-log-requests", action="store_true", default=False)
    group.add_argument("--kv-events-config", type=str, default=None, help="JSON config for KV cache event reporting")
    group.add_argument("--kv-transfer-config", type=str, default=None, help="JSON config for KV cache transfer")


def _add_jiuwen_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Jiuwen worker")
    group.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
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
    group.add_argument("--metrics-interval", type=float, default=5.0,
                        help="Interval in seconds for publishing worker metrics (default: 5.0)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jiuwen vLLM Worker")
    _add_vllm_args(parser)
    _add_jiuwen_args(parser)

    return parser.parse_args()

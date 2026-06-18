"""jiuwen-infer-worker CLI 入口。

支持两种启动方式：
  1. 配置文件：jiuwen-infer-worker --config worker.yaml
  2. 命令行参数：jiuwen-infer-worker --model /path/to/model --port 8010 ...
  3. 混合模式：jiuwen-infer-worker --config worker.yaml --port 8020  (CLI 覆盖 YAML)
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger("jiuwen.worker")


def _build_parser(*, model_required: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jiuwen vLLM Worker")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML configuration file",
    )

    from openjiuwentools.infer_router.vllm.args import _add_jiuwen_args, _add_vllm_args

    _add_vllm_args(parser, model_required=model_required)
    _add_jiuwen_args(parser)
    return parser


def _merge_config_and_cli(argv: list[str] | None = None) -> argparse.Namespace:
    """先加载 YAML 配置作为默认值，再用 CLI 参数覆盖。"""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    has_config = bool(pre_args.config)
    parser = _build_parser(model_required=not has_config)

    if has_config:
        from openjiuwentools.infer_router.vllm.worker_config import load_worker_config

        yaml_ns = load_worker_config(pre_args.config)
        defaults = vars(yaml_ns)
        parser.set_defaults(**defaults)

    args = parser.parse_args(argv)
    if args.model is None:
        parser.error("--model is required (via config file or CLI argument)")
    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    args = _merge_config_and_cli()

    logger.info("Starting Jiuwen vLLM Worker (model=%s, port=%d, mode=%s)",
                args.model, args.port, args.worker_mode)

    import uvloop

    from openjiuwentools.infer_router.vllm.main import worker

    uvloop.run(worker(args))


if __name__ == "__main__":
    main()

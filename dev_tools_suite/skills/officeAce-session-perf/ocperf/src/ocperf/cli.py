"""Minimal CLI: only `ocperf skill` (OfficeClaw session perf pipeline)."""

from __future__ import annotations

import argparse
import logging

from ocperf.errors import OcperfError
from ocperf.skill_cmd import register_skill_parser

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(prog="ocperf", description="OfficeClaw session performance reports")
    sub = parser.add_subparsers(dest="command", required=True)
    register_skill_parser(sub)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except OcperfError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

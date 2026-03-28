from __future__ import annotations

import os
import sys

from openjiuwen_plugin.handlers import COMMAND_HANDLERS
from openjiuwen_plugin.logging_config import setup_logging
from openjiuwen_plugin.parsers import build_plugin_parser


def main(argv: list[str] | None = None) -> int:
    setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
    argsv = list(argv) if argv is not None else sys.argv[1:]

    prog = "openjiuwen-plugin"
    if argv is None and sys.argv:
        prog = os.path.basename(sys.argv[0]) or prog
    parser = build_plugin_parser(prog)
    args = parser.parse_args(argsv)

    handler = COMMAND_HANDLERS.get(args.plugin_command)
    if not handler:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

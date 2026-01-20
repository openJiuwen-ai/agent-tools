#!/usr/bin/env python3
"""
Command Line Interface for lg2jiuwentool

Usage:
    python -m lg2jiuwentool.cli <source_file> [options]

Examples:
    python -m lg2jiuwentool.cli my_langgraph_agent.py
    python -m lg2jiuwentool.cli my_langgraph_agent.py -o ./output
    python -m lg2jiuwentool.cli my_langgraph_agent.py --name my_agent
"""

import argparse
import sys
from pathlib import Path

from .migrator import migrate, MigrationOptions


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        prog="lg2jiuwentool",
        description="Migrate LangGraph agents to openJiuwen framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s agent.py                    Migrate agent.py to current directory
  %(prog)s agent.py -o ./output        Migrate to ./output directory
  %(prog)s agent.py --name my_agent    Set output file name
  %(prog)s agent.py --no-report        Skip generating migration report

For more information, see: https://github.com/your-repo/lg2jiuwentool
        """
    )

    parser.add_argument(
        "source",
        type=str,
        help="Path to the LangGraph source file"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=".",
        help="Output directory (default: current directory)"
    )

    parser.add_argument(
        "-n", "--name",
        type=str,
        default=None,
        help="Output file name (without .py extension)"
    )

    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Don't generate migration report"
    )

    parser.add_argument(
        "--no-comments",
        action="store_true",
        help="Don't preserve original comments"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show verbose output"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )

    return parser


def main(args=None):
    """Main entry point"""
    parser = create_parser()
    parsed = parser.parse_args(args)

    # Validate source file
    source_path = Path(parsed.source)
    if not source_path.exists():
        print(f"Error: Source file not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    if not source_path.suffix == ".py":
        print(f"Warning: Source file does not have .py extension: {source_path}")

    # Create options
    options = MigrationOptions(
        preserve_comments=not parsed.no_comments,
        include_report=not parsed.no_report,
        output_name=parsed.name,
    )

    # Show start message
    if parsed.verbose:
        print(f"Migrating: {source_path}")
        print(f"Output directory: {parsed.output}")

    # Run migration
    result = migrate(
        source_path=str(source_path),
        output_dir=parsed.output,
        options=options
    )

    # Show result
    if result.success:
        print("Migration completed successfully!")
        print()
        print("Generated files:")
        for f in result.generated_files:
            print(f"  - {f}")

        if result.warnings:
            print()
            print("Warnings:")
            for w in result.warnings:
                print(f"  ! {w}")

        if result.manual_tasks:
            print()
            print("Manual tasks required:")
            for t in result.manual_tasks:
                print(f"  [ ] {t}")

        if parsed.verbose:
            print()
            print(result.report.to_string())

    else:
        print(f"Migration failed: {result.error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

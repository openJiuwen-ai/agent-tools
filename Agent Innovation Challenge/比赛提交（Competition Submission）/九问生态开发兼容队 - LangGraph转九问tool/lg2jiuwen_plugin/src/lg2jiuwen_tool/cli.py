#!/usr/bin/env python3
"""
LG2Jiuwen 命令行接口

将 LangGraph 代码迁移到 openJiuwen 框架

Usage:
    python -m lg2jiuwen_tool <source> [options]
    lg2jiuwen <source> [options]

Examples:
    lg2jiuwen my_agent.py
    lg2jiuwen my_agent.py -o ./output
    lg2jiuwen ./my_project/ -o ./output --use-ai
"""

import argparse
import sys
from pathlib import Path

from .service import migrate_new, MigrationOptions, MigrationResult


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        prog="lg2jiuwen",
        description="将 LangGraph 代码迁移到 openJiuwen 框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s agent.py                    迁移单个文件到当前目录
  %(prog)s agent.py -o ./output        迁移到指定目录
  %(prog)s ./project/ -o ./output      迁移整个项目
  %(prog)s agent.py --use-ai           启用 AI 处理未识别代码
  %(prog)s agent.py --no-report        不生成迁移报告

更多信息请访问: https://github.com/openjiuwen/lg2jiuwen
        """
    )

    parser.add_argument(
        "source",
        type=str,
        help="源文件或目录路径"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./output",
        help="输出目录 (默认: ./output)"
    )

    parser.add_argument(
        "--use-ai",
        action="store_true",
        help="启用 AI 处理规则无法转换的代码"
    )

    parser.add_argument(
        "--no-report",
        action="store_true",
        help="不生成迁移报告"
    )

    parser.add_argument(
        "--no-comments",
        action="store_true",
        help="不保留原始注释"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细输出"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )

    return parser


def print_result(result: MigrationResult, verbose: bool = False) -> None:
    """打印迁移结果"""
    if result.success:
        print("✓ 迁移成功!")
        print()

        if result.generated_files:
            print("生成的文件:")
            for f in result.generated_files:
                print(f"  - {f}")
            print()

        print(f"统计:")
        print(f"  - 规则处理: {result.rule_count} 项")
        print(f"  - AI 处理: {result.ai_count} 项")

        if verbose and result.report:
            print()
            print("=" * 60)
            print("迁移报告:")
            print("=" * 60)
            print(result.report)

    else:
        print("✗ 迁移失败!", file=sys.stderr)
        if result.errors:
            print()
            print("错误信息:", file=sys.stderr)
            for err in result.errors:
                print(f"  - {err}", file=sys.stderr)


def main(args=None) -> int:
    """主入口"""
    parser = create_parser()
    parsed = parser.parse_args(args)

    # 验证源路径
    source_path = Path(parsed.source)
    if not source_path.exists():
        print(f"错误: 源路径不存在: {source_path}", file=sys.stderr)
        return 1

    # 检查是否为 Python 文件或目录
    if source_path.is_file() and source_path.suffix != ".py":
        print(f"警告: 源文件不是 Python 文件: {source_path}")

    # 创建选项
    options = MigrationOptions(
        use_ai=parsed.use_ai,
        preserve_comments=not parsed.no_comments,
        include_report=not parsed.no_report,
        verbose=parsed.verbose
    )

    # 显示开始信息
    if parsed.verbose:
        print(f"源路径: {source_path}")
        print(f"输出目录: {parsed.output}")
        print(f"使用 AI: {'是' if options.use_ai else '否'}")
        print()

    # 执行迁移
    try:
        result = migrate_new(
            source_path=str(source_path),
            output_dir=parsed.output,
            options=options
        )

        # 显示结果
        print_result(result, verbose=parsed.verbose)

        return 0 if result.success else 1

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        if parsed.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

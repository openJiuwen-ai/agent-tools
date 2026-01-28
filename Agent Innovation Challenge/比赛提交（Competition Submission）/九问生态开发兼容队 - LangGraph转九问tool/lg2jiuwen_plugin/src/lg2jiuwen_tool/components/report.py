"""
报告生成组件

生成迁移报告
"""

from datetime import datetime
from typing import Any, Dict, List

from openjiuwen.core.component.base import WorkflowComponent
from openjiuwen.core.runtime.base import ComponentExecutable, Input, Output
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.context_engine.base import Context

from ..workflow.state import ExtractionResult
from ..ir.models import MigrationIR


class ReportComp(WorkflowComponent, ComponentExecutable):
    """
    报告生成组件

    功能：
    - 生成详细的迁移报告
    - 包含转换统计、特性支持、手动检查项
    """

    async def invoke(
        self,
        inputs: Input,
        runtime: Runtime,
        context: Context
    ) -> Output:
        # 从 inputs 获取（通过 transformer 传入）
        extraction_result: ExtractionResult = inputs.get("extraction_result")
        migration_ir: MigrationIR = inputs.get("migration_ir")
        generated_files: List[str] = inputs.get("generated_files", [])

        report = self._generate_report(
            extraction_result,
            generated_files,
            migration_ir
        )

        return {
            "report": report,
            "generated_files": generated_files
        }

    def _generate_report(
        self,
        result: ExtractionResult,
        generated_files: List[str],
        migration_ir: MigrationIR
    ) -> str:
        """生成迁移报告"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        sections = [
            self._gen_header(now),
            self._gen_summary(result, migration_ir),
            self._gen_conversion_stats(result),
            self._gen_nodes_detail(result),
            self._gen_edges_detail(result),
            self._gen_tools_detail(result),
            self._gen_generated_files(generated_files),
            self._gen_manual_checks(result),
            self._gen_footer()
        ]

        return "\n\n".join(sections)

    def _gen_header(self, timestamp: str) -> str:
        """生成报告头"""
        return f"""=== LangGraph to openJiuwen 迁移报告 ===
生成时间: {timestamp}
工具版本: LG2Jiuwen v2.0"""

    def _gen_summary(self, result: ExtractionResult, migration_ir: MigrationIR) -> str:
        """生成摘要"""
        agent_name = migration_ir.agent_ir.name if migration_ir else "Unknown"
        total_nodes = len(result.nodes)
        total_edges = len(result.edges)
        total_tools = len(result.tools)

        return f"""## 摘要

- Agent 名称: {agent_name}
- 状态类: {result.state_class_name or "未检测到"}
- 入口节点: {result.entry_point or "未检测到"}
- 节点数量: {total_nodes}
- 边数量: {total_edges}
- 工具数量: {total_tools}"""

    def _gen_conversion_stats(self, result: ExtractionResult) -> str:
        """生成转换统计"""
        total = result.rule_count + result.ai_count
        rule_pct = (result.rule_count / total * 100) if total > 0 else 0
        ai_pct = (result.ai_count / total * 100) if total > 0 else 0

        return f"""## 转换统计

| 处理方式 | 数量 | 占比 |
|---------|------|------|
| 规则处理 | {result.rule_count} | {rule_pct:.1f}% |
| AI 处理 | {result.ai_count} | {ai_pct:.1f}% |
| **总计** | **{total}** | **100%** |"""

    def _gen_nodes_detail(self, result: ExtractionResult) -> str:
        """生成节点详情"""
        lines = ["## 节点详情", "", "| 节点名 | 转换方式 | 输入字段 | 输出字段 |", "|--------|---------|---------|---------|"]

        for node in result.nodes:
            inputs = ", ".join(node.inputs) if node.inputs else "-"
            outputs = ", ".join(node.outputs) if node.outputs else "-"
            lines.append(f"| {node.name} | {node.conversion_source} | {inputs} | {outputs} |")

        return "\n".join(lines)

    def _gen_edges_detail(self, result: ExtractionResult) -> str:
        """生成边详情"""
        lines = ["## 边详情", "", "| 源节点 | 目标节点 | 类型 |", "|--------|---------|------|"]

        for edge in result.edges:
            edge_type = "条件边" if edge.is_conditional else "普通边"
            target = edge.target if edge.target else str(edge.condition_map)
            lines.append(f"| {edge.source} | {target} | {edge_type} |")

        return "\n".join(lines)

    def _gen_tools_detail(self, result: ExtractionResult) -> str:
        """生成工具详情"""
        if not result.tools:
            return "## 工具详情\n\n无工具定义"

        lines = ["## 工具详情", "", "| 工具名 | 描述 | 参数 |", "|--------|------|------|"]

        for tool in result.tools:
            params = ", ".join(p["name"] for p in tool.parameters) if tool.parameters else "-"
            desc = (tool.description[:30] + "...") if len(tool.description) > 30 else tool.description
            lines.append(f"| {tool.name} | {desc} | {params} |")

        return "\n".join(lines)

    def _gen_generated_files(self, files: List[str]) -> str:
        """生成文件列表"""
        lines = ["## 生成的文件", ""]

        for f in files:
            lines.append(f"- {f}")

        if not files:
            lines.append("- 无")

        return "\n".join(lines)

    def _gen_manual_checks(self, result: ExtractionResult) -> str:
        """生成手动检查项"""
        checks = [
            "[ ] 验证 LLM 配置（API Key, API Base）",
            "[ ] 测试条件路由逻辑",
            "[ ] 确认异步运行环境",
            "[ ] 检查状态字段映射",
        ]

        # 如果有 AI 处理的节点，添加额外检查项
        ai_nodes = [n for n in result.nodes if n.conversion_source == "ai"]
        if ai_nodes:
            checks.append(f"[ ] 人工审核 AI 转换的 {len(ai_nodes)} 个节点")

        # 如果有工具，添加工具检查
        if result.tools:
            checks.append("[ ] 验证工具函数的参数和返回值")

        lines = ["## 手动检查项", ""]
        lines.extend(checks)

        return "\n".join(lines)

    def _gen_footer(self) -> str:
        """生成报告尾"""
        return """---
由 LG2Jiuwen 自动生成
如有问题请联系开发团队"""

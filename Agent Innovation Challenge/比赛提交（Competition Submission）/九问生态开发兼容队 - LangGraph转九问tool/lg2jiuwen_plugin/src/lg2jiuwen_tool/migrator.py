"""
Main Migrator Module

Orchestrates the complete migration from LangGraph to openJiuwen.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from enum import Enum

from .parser import LangGraphParser
from .generator import OpenJiuwenGenerator
from .ir_models import (
    AgentIR,
    WorkflowIR,
    WorkflowNodeIR,
    WorkflowNodeType,
    WorkflowEdgeIR,
    ToolIR,
    ParamInfo,
    ParseResult,
    LLMConfigIR,
)


@dataclass
class MigrationOptions:
    """Options for the migration process"""
    # Whether to preserve original comments
    preserve_comments: bool = True
    # Whether to generate type hints
    generate_type_hints: bool = True
    # Whether to include migration report
    include_report: bool = True
    # Whether to include IR (Intermediate Representation) dump
    include_ir: bool = True
    # Output file name (without extension)
    output_name: Optional[str] = None
    # Whether to format the output code
    format_code: bool = True


@dataclass
class MigrationReport:
    """Report of the migration process"""
    source_file: str
    output_file: str
    nodes_converted: int
    edges_converted: int
    tools_converted: int
    warnings: List[str] = field(default_factory=list)
    manual_tasks: List[str] = field(default_factory=list)
    summary: str = ""

    def to_string(self) -> str:
        """Convert report to formatted string"""
        lines = [
            "=" * 60,
            "Migration Report",
            "=" * 60,
            f"Source: {self.source_file}",
            f"Output: {self.output_file}",
            "",
            "Statistics:",
            f"  - Nodes converted: {self.nodes_converted}",
            f"  - Edges converted: {self.edges_converted}",
            f"  - Tools converted: {self.tools_converted}",
            "",
        ]

        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
            lines.append("")

        if self.manual_tasks:
            lines.append("Manual Tasks Required:")
            for t in self.manual_tasks:
                lines.append(f"  [ ] {t}")
            lines.append("")

        lines.append(self.summary)
        lines.append("=" * 60)

        return "\n".join(lines)


@dataclass
class MigrationResult:
    """Result of the migration process"""
    success: bool
    generated_files: List[str]
    report: MigrationReport
    warnings: List[str] = field(default_factory=list)
    manual_tasks: List[str] = field(default_factory=list)
    error: Optional[str] = None


def serialize_ir(agent_ir: AgentIR, parse_result: ParseResult) -> Dict[str, Any]:
    """Serialize the IR to a JSON-serializable dictionary.

    This provides a complete view of the intermediate representation
    extracted from the source LangGraph code.
    """
    def serialize_obj(obj):
        """Recursively serialize an object to JSON-compatible format"""
        if obj is None:
            return None
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [serialize_obj(item) for item in obj]
        if isinstance(obj, dict):
            return {k: serialize_obj(v) for k, v in obj.items()}
        if hasattr(obj, '__dataclass_fields__'):
            return {k: serialize_obj(v) for k, v in asdict(obj).items()}
        if hasattr(obj, '__dict__'):
            return {k: serialize_obj(v) for k, v in obj.__dict__.items()
                    if not k.startswith('_')}
        return str(obj)

    ir_data = {
        "_meta": {
            "description": "Intermediate Representation (IR) extracted during LangGraph to openJiuwen migration",
            "source_framework": "langgraph",
            "target_framework": "openjiuwen",
        },
        "parse_result": {
            "graph_name": parse_result.graph_name,
            "state_class_name": parse_result.state_class_name,
            "state_fields": [
                {
                    "name": f.name,
                    "type_hint": f.type_hint,
                    "has_aggregator": f.has_aggregator,
                    "aggregator": f.aggregator,
                }
                for f in parse_result.state_fields
            ],
            "node_functions": {
                name: {
                    "name": info.name,
                    "docstring": info.docstring,
                    "params": [{"name": p.name, "type_hint": p.type_hint} for p in info.params],
                    "state_accesses": [
                        {"key": a.key, "access_type": a.access_type, "default_value": a.default_value}
                        for a in info.state_accesses
                    ],
                    "return_keys": info.return_keys,
                    "is_conditional": info.is_conditional,
                    "condition_returns": info.condition_returns,
                    "uses_llm": getattr(info, 'uses_llm', False),
                    "body": info.body,
                }
                for name, info in parse_result.node_functions.items()
            },
            "conditional_functions": {
                name: {
                    "name": info.name,
                    "docstring": info.docstring,
                    "state_accesses": [
                        {"key": a.key, "access_type": a.access_type, "default_value": a.default_value}
                        for a in info.state_accesses
                    ],
                    "condition_returns": info.condition_returns,
                    "body": info.body,
                }
                for name, info in parse_result.conditional_functions.items()
            },
            "edges": [
                {"source": e.source, "target": e.target}
                for e in parse_result.edges
            ],
            "conditional_edges": [
                {
                    "source_node": e.source_node,
                    "condition_function": e.condition_function,
                    "mapping": e.mapping,
                }
                for e in parse_result.conditional_edges
            ],
            "entry_point": parse_result.entry_point,
            "llm_config": serialize_obj(parse_result.llm_config) if parse_result.llm_config else None,
        },
        "agent_ir": {
            "name": agent_ir.name,
            "description": agent_ir.description,
            "agent_type": agent_ir.agent_type,
            "llm_config": serialize_obj(agent_ir.llm_config) if agent_ir.llm_config else None,
            "state_fields": [
                {
                    "name": f.name,
                    "type_hint": f.type_hint,
                    "has_aggregator": f.has_aggregator,
                    "aggregator": f.aggregator,
                }
                for f in agent_ir.state_fields
            ],
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "params": [{"name": p.name, "type_hint": p.type_hint} for p in t.params],
                    "return_type": t.return_type,
                    "function_body": t.function_body,
                }
                for t in agent_ir.tools
            ],
        },
        "workflow_ir": None,
    }

    # Serialize workflow IR if present
    if agent_ir.workflow:
        ir_data["workflow_ir"] = {
            "name": agent_ir.workflow.name,
            "entry_node": agent_ir.workflow.entry_node,
            "nodes": [
                {
                    "id": n.id,
                    "node_type": n.node_type.value,
                    "name": n.name,
                    "config": serialize_obj(n.config),
                }
                for n in agent_ir.workflow.nodes
            ],
            "edges": [
                {
                    "source_node": e.source_node,
                    "target_node": e.target_node,
                    "condition": e.condition,
                    "branch_id": e.branch_id,
                }
                for e in agent_ir.workflow.edges
            ],
        }

    return ir_data


class IRBuilder:
    """Builds IR from parsed results"""

    def build(self, parse_result: ParseResult) -> AgentIR:
        """Build AgentIR from ParseResult"""
        # Build workflow IR
        workflow_ir = self._build_workflow_ir(parse_result)

        # Build tool IRs
        tools_ir = [self._build_tool_ir(t) for t in parse_result.tools]

        # Determine agent type
        agent_type = self._determine_agent_type(parse_result)

        return AgentIR(
            name=parse_result.graph_name or "migrated_agent",
            description=f"Migrated from LangGraph",
            agent_type=agent_type,
            llm_config=parse_result.llm_config,
            tools=tools_ir,
            workflow=workflow_ir,
            state_fields=parse_result.state_fields,
            source_framework="langgraph",
            metadata={
                "node_functions": {
                    name: {
                        "state_accesses": [a.__dict__ for a in info.state_accesses],
                        "is_conditional": info.is_conditional,
                        "condition_returns": info.condition_returns,
                        "return_keys": info.return_keys,
                        "docstring": info.docstring,
                    }
                    for name, info in parse_result.node_functions.items()
                },
                "conditional_functions": {
                    name: {
                        "state_accesses": [a.__dict__ for a in info.state_accesses],
                        "is_conditional": info.is_conditional,
                        "condition_returns": info.condition_returns,
                        "return_keys": info.return_keys,
                        "docstring": info.docstring,
                        "body_statements": getattr(info, 'body_statements', []),
                        "return_statement": getattr(info, 'return_statement', None),
                        "body": info.body,
                    }
                    for name, info in parse_result.conditional_functions.items()
                },
                "conditional_edges": [
                    {
                        "source_node": e.source_node,
                        "condition_function": e.condition_function,
                        "mapping": e.mapping
                    }
                    for e in parse_result.conditional_edges
                ],
                "global_variables": [
                    {"name": v.name, "value": v.value, "type_hint": v.type_hint}
                    for v in parse_result.global_variables
                ],
                "import_statements": parse_result.import_statements,
                "example_inputs": parse_result.example_inputs,
            }
        )

    def _build_workflow_ir(self, parse_result: ParseResult) -> WorkflowIR:
        """Build WorkflowIR from parse result"""
        nodes = []
        edges = []

        # Add Start node
        nodes.append(WorkflowNodeIR(
            id="start",
            node_type=WorkflowNodeType.START,
            name="Start",
            config={"inputs": [f.name for f in parse_result.state_fields]}
        ))

        # Convert node functions to nodes
        for name, func_info in parse_result.node_functions.items():
            # Check if this node has conditional edges
            is_conditional = any(
                e.source_node == name for e in parse_result.conditional_edges
            )

            # Get condition returns if it's a conditional node
            condition_returns = []
            if is_conditional:
                for e in parse_result.conditional_edges:
                    if e.source_node == name:
                        # Get returns from the condition function
                        cond_func = parse_result.conditional_functions.get(e.condition_function)
                        if cond_func:
                            condition_returns = cond_func.condition_returns
                        break

            # Determine node type: LLM nodes take precedence
            uses_llm = getattr(func_info, 'uses_llm', False)
            if uses_llm:
                node_type = WorkflowNodeType.LLM
            elif is_conditional:
                node_type = WorkflowNodeType.CONDITIONAL
            else:
                node_type = WorkflowNodeType.FUNCTION

            nodes.append(WorkflowNodeIR(
                id=name,
                node_type=node_type,
                name=name,
                config={
                    "function_body": func_info.body,
                    "state_accesses": [a.__dict__ for a in func_info.state_accesses],
                    "return_keys": func_info.return_keys,
                    "is_conditional": is_conditional,
                    "condition_returns": condition_returns,
                    "docstring": func_info.docstring,
                    # New fields for better code generation
                    "body_statements": getattr(func_info, 'body_statements', []),
                    "return_statement": getattr(func_info, 'return_statement', None),
                    # LLM usage flag
                    "uses_llm": uses_llm,
                }
            ))

        # Add End node
        nodes.append(WorkflowNodeIR(
            id="end",
            node_type=WorkflowNodeType.END,
            name="End"
        ))

        # Convert edges
        # Add entry edge
        if parse_result.entry_point:
            edges.append(WorkflowEdgeIR(
                source_node="start",
                target_node=parse_result.entry_point
            ))

        # Add regular edges
        for edge in parse_result.edges:
            target = edge.target if edge.target != "END" else "end"
            edges.append(WorkflowEdgeIR(
                source_node=edge.source,
                target_node=target
            ))

        # Add conditional edges
        for cond_edge in parse_result.conditional_edges:
            for return_val, target in cond_edge.mapping.items():
                target_node = target if target != "END" else "end"
                edges.append(WorkflowEdgeIR(
                    source_node=cond_edge.source_node,
                    target_node=target_node,
                    condition=f"${{{cond_edge.source_node}.route}} == '{return_val}'",
                    branch_id=return_val
                ))

        return WorkflowIR(
            name=parse_result.graph_name or "workflow",
            nodes=nodes,
            edges=edges,
            entry_node="start"
        )

    def _build_tool_ir(self, tool_info) -> ToolIR:
        """Build ToolIR from ToolInfo"""
        return ToolIR(
            name=tool_info.name,
            description=tool_info.description,
            params=tool_info.params,
            return_type=tool_info.return_type,
            function_body=tool_info.function_body
        )

    def _determine_agent_type(self, parse_result: ParseResult) -> str:
        """Determine the type of agent"""
        # Check for ReAct pattern
        has_tools = len(parse_result.tools) > 0
        has_llm = parse_result.llm_config is not None

        if has_tools and has_llm:
            return "react"
        elif has_llm:
            return "llm"
        else:
            return "workflow"


def migrate(
    source_path: str,
    output_dir: str,
    options: MigrationOptions = None
) -> MigrationResult:
    """
    Migrate LangGraph code to openJiuwen.

    Args:
        source_path: Path to the LangGraph source file
        output_dir: Directory to write the output
        options: Migration options

    Returns:
        MigrationResult with details of the migration
    """
    options = options or MigrationOptions()

    try:
        # 1. Read source file
        source_path = Path(source_path)
        if not source_path.exists():
            return MigrationResult(
                success=False,
                generated_files=[],
                report=MigrationReport(
                    source_file=str(source_path),
                    output_file="",
                    nodes_converted=0,
                    edges_converted=0,
                    tools_converted=0,
                    summary="Source file not found"
                ),
                error=f"Source file not found: {source_path}"
            )

        with open(source_path, "r", encoding="utf-8") as f:
            source_code = f.read()

        # 2. Parse
        parser = LangGraphParser()
        parse_result = parser.parse(source_code)

        # 3. Build IR
        ir_builder = IRBuilder()
        agent_ir = ir_builder.build(parse_result)

        # 4. Generate code
        generator = OpenJiuwenGenerator()
        output_name = options.output_name or source_path.stem + "_openjiuwen"
        generated_code = generator.generate(agent_ir, output_name)

        # 5. Write output
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{output_name}.py"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(generated_code)

        generated_files = [str(output_file)]

        # 6. Generate report
        warnings = generator.warnings + _generate_warnings(parse_result)
        manual_tasks = generator.manual_tasks + _generate_manual_tasks(parse_result)

        report = MigrationReport(
            source_file=str(source_path),
            output_file=str(output_file),
            nodes_converted=len(parse_result.node_functions),
            edges_converted=len(parse_result.edges) + len(parse_result.conditional_edges),
            tools_converted=len(parse_result.tools),
            warnings=warnings,
            manual_tasks=manual_tasks,
            summary="Migration completed successfully!"
        )

        # Write report if requested
        if options.include_report:
            report_file = output_dir / f"{output_name}_report.txt"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report.to_string())
            generated_files.append(str(report_file))

        # Write IR (Intermediate Representation) if requested
        if options.include_ir:
            ir_file = output_dir / f"{output_name}_ir.json"
            ir_data = serialize_ir(agent_ir, parse_result)
            with open(ir_file, "w", encoding="utf-8") as f:
                json.dump(ir_data, f, indent=2, ensure_ascii=False)
            generated_files.append(str(ir_file))

        return MigrationResult(
            success=True,
            generated_files=generated_files,
            report=report,
            warnings=warnings,
            manual_tasks=manual_tasks
        )

    except Exception as e:
        return MigrationResult(
            success=False,
            generated_files=[],
            report=MigrationReport(
                source_file=str(source_path),
                output_file="",
                nodes_converted=0,
                edges_converted=0,
                tools_converted=0,
                summary=f"Migration failed: {str(e)}"
            ),
            error=str(e)
        )


def _generate_warnings(parse_result: ParseResult) -> List[str]:
    """Generate warnings based on parse result"""
    warnings = []

    # Check for state aggregators
    for field in parse_result.state_fields:
        if field.has_aggregator:
            warnings.append(
                f"State field '{field.name}' uses aggregator '{field.aggregator}'. "
                "openJiuwen doesn't have built-in aggregators - manual handling required."
            )

    # Check for complex conditional edges
    if len(parse_result.conditional_edges) > 1:
        warnings.append(
            "Multiple conditional edges detected. Review the generated BranchRouter configurations."
        )

    return warnings


def _generate_manual_tasks(parse_result: ParseResult) -> List[str]:
    """Generate list of manual tasks"""
    tasks = []

    # Review component logic
    tasks.append("Review and adapt component invoke() method implementations")

    # State aggregation
    if any(f.has_aggregator for f in parse_result.state_fields):
        tasks.append("Implement manual state aggregation logic for fields with aggregators")

    # Conditional routing
    if parse_result.conditional_edges:
        tasks.append("Verify BranchRouter conditions and target mappings")

    # LLM integration
    tasks.append("Configure LLM provider and model settings in component initializations")

    # Testing
    tasks.append("Test the migrated workflow with sample inputs")

    return tasks

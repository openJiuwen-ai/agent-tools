"""
迁移工作流定义

定义 LangGraph 到 openJiuwen 的迁移工作流
"""

from openjiuwen.core.workflow.base import Workflow
from openjiuwen.core.component.start_comp import Start
from openjiuwen.core.component.end_comp import End
from openjiuwen.core.runtime.state import ReadableStateLike

from ..components.project_detector import ProjectDetectorComp
from ..components.file_loader import FileLoaderComp
from ..components.ast_parser import ASTParserComp
from ..components.rule_extractor import RuleExtractorComp
from ..components.pending_check import PendingCheckComp, pending_router
from ..components.ai_semantic import AISemanticComp
from ..components.ir_builder import IRBuilderComp
from ..components.code_generator import CodeGeneratorComp
from ..components.report import ReportComp


# ==================== Transformer 定义 ====================

def _unwrap_state_value(value):
    """
    解包 openJiuwen 状态值

    openJiuwen 的 state.get() 可能返回 {"": actual_value} 格式，
    需要解包获取实际值
    """
    if isinstance(value, dict) and "" in value and len(value) == 1:
        return value[""]
    return value


def loader_inputs_transformer(state: ReadableStateLike):
    """FileLoader 输入转换器"""
    return {
        "file_list": _unwrap_state_value(state.get("detector.file_list")),
        "dependency_order": _unwrap_state_value(state.get("detector.dependency_order"))
    }


def parser_inputs_transformer(state: ReadableStateLike):
    """ASTParser 输入转换器"""
    file_contents = _unwrap_state_value(state.get("loader.file_contents"))
    dependency_order = _unwrap_state_value(state.get("loader.dependency_order"))

    return {
        "file_contents": file_contents,
        "dependency_order": dependency_order
    }


def extractor_inputs_transformer(state: ReadableStateLike):
    """RuleExtractor 输入转换器"""
    return {
        "ast_map": _unwrap_state_value(state.get("parser.ast_map")),
        "dependency_order": _unwrap_state_value(state.get("parser.dependency_order"))
    }


def checker_inputs_transformer(state: ReadableStateLike):
    """PendingCheck 输入转换器"""
    return {
        "extraction_result": _unwrap_state_value(state.get("extractor.extraction_result"))
    }


def ai_inputs_transformer(state: ReadableStateLike):
    """AISemantic 输入转换器"""
    return {
        "extraction_result": _unwrap_state_value(state.get("checker.extraction_result"))
    }


def ir_builder_inputs_transformer(state: ReadableStateLike):
    """IRBuilder 输入转换器 (从 AI 组件获取)"""
    return {
        "extraction_result": _unwrap_state_value(state.get("ai.extraction_result"))
    }


def ir_builder_direct_inputs_transformer(state: ReadableStateLike):
    """IRBuilder 输入转换器 (直接从 checker 获取)"""
    return {
        "extraction_result": _unwrap_state_value(state.get("checker.extraction_result"))
    }


def generator_inputs_transformer(state: ReadableStateLike):
    """CodeGenerator 输入转换器"""
    return {
        "agent_ir": _unwrap_state_value(state.get("ir_builder.agent_ir")),
        "workflow_ir": _unwrap_state_value(state.get("ir_builder.workflow_ir")),
        "migration_ir": _unwrap_state_value(state.get("ir_builder.migration_ir")),
        "output_dir": _unwrap_state_value(state.get("start.output_dir")),
        "is_multi_file": _unwrap_state_value(state.get("detector.is_multi_file")),
        "project_root": _unwrap_state_value(state.get("detector.project_root"))
    }


def generator_direct_inputs_transformer(state: ReadableStateLike):
    """CodeGenerator 输入转换器 (直接路径)"""
    return {
        "agent_ir": _unwrap_state_value(state.get("ir_builder_direct.agent_ir")),
        "workflow_ir": _unwrap_state_value(state.get("ir_builder_direct.workflow_ir")),
        "migration_ir": _unwrap_state_value(state.get("ir_builder_direct.migration_ir")),
        "output_dir": _unwrap_state_value(state.get("start.output_dir")),
        "is_multi_file": _unwrap_state_value(state.get("detector.is_multi_file")),
        "project_root": _unwrap_state_value(state.get("detector.project_root"))
    }


def reporter_inputs_transformer(state: ReadableStateLike):
    """Report 输入转换器"""
    return {
        "extraction_result": _unwrap_state_value(state.get("ir_builder.extraction_result")),
        "generated_files": _unwrap_state_value(state.get("generator.generated_files")),
        "migration_ir": _unwrap_state_value(state.get("ir_builder.migration_ir"))
    }


def reporter_direct_inputs_transformer(state: ReadableStateLike):
    """Report 输入转换器 (直接路径)"""
    return {
        "extraction_result": _unwrap_state_value(state.get("ir_builder_direct.extraction_result")),
        "generated_files": _unwrap_state_value(state.get("generator_direct.generated_files")),
        "migration_ir": _unwrap_state_value(state.get("ir_builder_direct.migration_ir"))
    }


# ==================== 工作流构建 ====================

def build_migration_workflow(llm=None) -> Workflow:
    """
    构建迁移工作流

    Args:
        llm: 可选的 LLM 实例，用于 AI 语义理解

    Returns:
        Workflow: 迁移工作流实例
    """
    workflow = Workflow()

    # ========== 设置起点 ==========
    workflow.set_start_comp(
        "start",
        Start(),
        inputs_schema={
            "source_path": "${source_path}",
            "output_dir": "${output_dir}"
        }
    )

    # ========== 项目检测 ==========
    workflow.add_workflow_comp(
        "detector",
        ProjectDetectorComp(),
        inputs_schema={
            "source_path": "${start.source_path}"
        }
    )

    # ========== 文件加载 ==========
    workflow.add_workflow_comp(
        "loader",
        FileLoaderComp(),
        inputs_transformer=loader_inputs_transformer
    )

    # ========== AST 解析 ==========
    workflow.add_workflow_comp(
        "parser",
        ASTParserComp(),
        inputs_transformer=parser_inputs_transformer
    )

    # ========== 规则提取 ==========
    workflow.add_workflow_comp(
        "extractor",
        RuleExtractorComp(),
        inputs_transformer=extractor_inputs_transformer
    )

    # ========== 待处理检查 ==========
    workflow.add_workflow_comp(
        "checker",
        PendingCheckComp(),
        inputs_transformer=checker_inputs_transformer
    )

    # ========== AI 语义理解（条件触发）==========
    workflow.add_workflow_comp(
        "ai",
        AISemanticComp(llm=llm),
        inputs_transformer=ai_inputs_transformer
    )

    # ========== IR 构建 ==========
    workflow.add_workflow_comp(
        "ir_builder",
        IRBuilderComp(),
        inputs_transformer=ir_builder_inputs_transformer
    )

    # 当没有 pending 时，直接从 checker 到 ir_builder
    workflow.add_workflow_comp(
        "ir_builder_direct",
        IRBuilderComp(),
        inputs_transformer=ir_builder_direct_inputs_transformer
    )

    # ========== 代码生成 ==========
    workflow.add_workflow_comp(
        "generator",
        CodeGeneratorComp(),
        inputs_transformer=generator_inputs_transformer
    )

    workflow.add_workflow_comp(
        "generator_direct",
        CodeGeneratorComp(),
        inputs_transformer=generator_direct_inputs_transformer
    )

    # ========== 报告生成 ==========
    workflow.add_workflow_comp(
        "reporter",
        ReportComp(),
        inputs_transformer=reporter_inputs_transformer
    )

    workflow.add_workflow_comp(
        "reporter_direct",
        ReportComp(),
        inputs_transformer=reporter_direct_inputs_transformer
    )

    # ========== 设置终点 ==========
    workflow.set_end_comp(
        "end",
        End(),
        inputs_schema={
            "generated_files": "${reporter.generated_files}",
            "report": "${reporter.report}"
        }
    )

    workflow.set_end_comp(
        "end_direct",
        End(),
        inputs_schema={
            "generated_files": "${reporter_direct.generated_files}",
            "report": "${reporter_direct.report}"
        }
    )

    # ========== 添加连接 ==========
    workflow.add_connection("start", "detector")
    workflow.add_connection("detector", "loader")
    workflow.add_connection("loader", "parser")
    workflow.add_connection("parser", "extractor")
    workflow.add_connection("extractor", "checker")

    # 条件路由：是否需要 AI 处理
    workflow.add_conditional_connection("checker", router=pending_router)

    # AI 处理后的路径
    workflow.add_connection("ai", "ir_builder")
    workflow.add_connection("ir_builder", "generator")
    workflow.add_connection("generator", "reporter")
    workflow.add_connection("reporter", "end")

    # 直接处理的路径
    workflow.add_connection("ir_builder_direct", "generator_direct")
    workflow.add_connection("generator_direct", "reporter_direct")
    workflow.add_connection("reporter_direct", "end_direct")

    return workflow


def build_simple_migration_workflow() -> Workflow:
    """
    构建简化版迁移工作流（不使用 AI）

    Returns:
        Workflow: 简化版迁移工作流实例
    """
    workflow = Workflow()

    # 起点
    workflow.set_start_comp(
        "start",
        Start(),
        inputs_schema={
            "source_path": "${source_path}",
            "output_dir": "${output_dir}"
        }
    )

    # 项目检测
    workflow.add_workflow_comp(
        "detector",
        ProjectDetectorComp(),
        inputs_schema={"source_path": "${start.source_path}"}
    )

    # 文件加载
    workflow.add_workflow_comp(
        "loader",
        FileLoaderComp(),
        inputs_transformer=loader_inputs_transformer
    )

    # AST 解析
    workflow.add_workflow_comp(
        "parser",
        ASTParserComp(),
        inputs_transformer=parser_inputs_transformer
    )

    # 规则提取
    workflow.add_workflow_comp(
        "extractor",
        RuleExtractorComp(),
        inputs_transformer=extractor_inputs_transformer
    )

    # IR 构建 - 直接从 extractor 获取
    def simple_ir_builder_inputs_transformer(state: ReadableStateLike):
        return {
            "extraction_result": _unwrap_state_value(state.get("extractor.extraction_result"))
        }

    workflow.add_workflow_comp(
        "ir_builder",
        IRBuilderComp(),
        inputs_transformer=simple_ir_builder_inputs_transformer
    )

    # 代码生成
    def simple_generator_inputs_transformer(state: ReadableStateLike):
        return {
            "agent_ir": _unwrap_state_value(state.get("ir_builder.agent_ir")),
            "workflow_ir": _unwrap_state_value(state.get("ir_builder.workflow_ir")),
            "migration_ir": _unwrap_state_value(state.get("ir_builder.migration_ir")),
            "output_dir": _unwrap_state_value(state.get("start.output_dir")),
            "is_multi_file": _unwrap_state_value(state.get("detector.is_multi_file")),
            "project_root": _unwrap_state_value(state.get("detector.project_root"))
        }

    workflow.add_workflow_comp(
        "generator",
        CodeGeneratorComp(),
        inputs_transformer=simple_generator_inputs_transformer
    )

    # 报告生成
    def simple_reporter_inputs_transformer(state: ReadableStateLike):
        return {
            "extraction_result": _unwrap_state_value(state.get("ir_builder.extraction_result")),
            "generated_files": _unwrap_state_value(state.get("generator.generated_files")),
            "migration_ir": _unwrap_state_value(state.get("ir_builder.migration_ir"))
        }

    workflow.add_workflow_comp(
        "reporter",
        ReportComp(),
        inputs_transformer=simple_reporter_inputs_transformer
    )

    # 终点
    workflow.set_end_comp(
        "end",
        End(),
        inputs_schema={
            "generated_files": "${reporter.generated_files}",
            "report": "${reporter.report}"
        }
    )

    # 连接
    workflow.add_connection("start", "detector")
    workflow.add_connection("detector", "loader")
    workflow.add_connection("loader", "parser")
    workflow.add_connection("parser", "extractor")
    workflow.add_connection("extractor", "ir_builder")
    workflow.add_connection("ir_builder", "generator")
    workflow.add_connection("generator", "reporter")
    workflow.add_connection("reporter", "end")

    return workflow

"""
状态和数据模型定义

定义迁移工作流中使用的所有数据结构
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PendingType(Enum):
    """待处理项类型"""
    NODE_BODY = "node_body"              # 节点函数体转换
    CONDITIONAL = "conditional"          # 条件路由逻辑
    TOOL_BODY = "tool_body"              # 工具函数体
    COMPLEX_EXPR = "complex_expr"        # 复杂表达式


@dataclass
class PendingItem:
    """
    待 AI 处理的项

    当规则无法处理某段代码时，生成 PendingItem 交给 AI 处理
    """
    id: str                              # 唯一标识，如 "file.py:func_name"
    pending_type: PendingType            # 待处理类型
    source_code: str                     # 原始代码
    context: Dict[str, Any]              # 上下文（状态字段、可用工具等）
    question: str                        # 给 AI 的具体问题
    location: str                        # 位置信息 (file:line)


@dataclass
class ConvertedNode:
    """
    已转换的节点

    无论是规则还是 AI 处理，最终都输出 ConvertedNode
    """
    name: str                            # 节点名
    original_code: str                   # 原始代码（用于报告）
    converted_body: str                  # 已转换的函数体代码
    inputs: List[str]                    # 输入字段列表
    outputs: List[str]                   # 输出字段列表
    conversion_source: str               # "rule" 或 "ai"
    docstring: Optional[str] = None      # 文档字符串


@dataclass
class StateField:
    """状态字段定义"""
    name: str                            # 字段名
    type_hint: str                       # 类型提示
    default: Optional[str] = None        # 默认值


@dataclass
class EdgeInfo:
    """边信息"""
    source: str                          # 源节点
    target: str                          # 目标节点
    is_conditional: bool = False         # 是否为条件边
    condition_func: Optional[str] = None # 条件函数名
    condition_func_code: Optional[str] = None  # 条件函数原始代码
    condition_map: Optional[Dict[str, str]] = None  # 条件映射


@dataclass
class ToolInfo:
    """工具信息"""
    name: str                            # 工具名
    original_code: str = ""              # 原始代码
    func_name: Optional[str] = None      # 函数名（当使用 Tool() 类时）
    converted_code: Optional[str] = None # 转换后的代码
    description: Optional[str] = None    # 描述
    parameters: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class LLMConfig:
    """LLM 配置信息"""
    var_name: str                        # 变量名
    model_class: str                     # 模型类名 (如 ChatOpenAI)
    model_name: Optional[str] = None     # 模型名称 (如 gpt-4)
    temperature: Optional[float] = None  # 温度参数
    other_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """
    提取结果

    包含已成功转换的内容和待处理项
    """
    # 已完成转换的内容
    states: List[StateField] = field(default_factory=list)
    nodes: List[ConvertedNode] = field(default_factory=list)
    edges: List[EdgeInfo] = field(default_factory=list)
    tools: List[ToolInfo] = field(default_factory=list)
    llm_configs: List[LLMConfig] = field(default_factory=list)
    global_vars: List[str] = field(default_factory=list)  # 全局变量定义
    tool_related_vars: List[str] = field(default_factory=list)  # 工具相关变量（如 tool_map）
    tool_map_var_name: Optional[str] = None  # 工具映射变量名（从源代码提取，如 tool_map、tools 等）

    # 待处理项（规则失败的）
    pending_items: List[PendingItem] = field(default_factory=list)

    # 其他提取信息
    entry_point: Optional[str] = None    # 入口节点
    graph_name: Optional[str] = None     # 图名称
    state_class_name: Optional[str] = None  # 状态类名
    imports: List[str] = field(default_factory=list)  # 原始导入语句
    initial_inputs: Dict[str, Any] = field(default_factory=dict)  # 初始输入（从 invoke 调用提取）
    example_inputs: Dict[str, Any] = field(default_factory=dict)  # 示例输入（从 main 函数提取）

    # 统计
    rule_count: int = 0                  # 规则处理数量
    ai_count: int = 0                    # AI处理数量

    def has_pending(self) -> bool:
        """是否有待处理项"""
        return len(self.pending_items) > 0

    def get_pending_summary(self) -> Dict[str, int]:
        """获取待处理项统计"""
        summary: Dict[str, int] = {}
        for item in self.pending_items:
            key = item.pending_type.value
            summary[key] = summary.get(key, 0) + 1
        return summary

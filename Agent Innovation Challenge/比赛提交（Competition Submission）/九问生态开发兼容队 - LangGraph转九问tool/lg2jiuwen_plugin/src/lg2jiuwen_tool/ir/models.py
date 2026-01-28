"""
中间表示 (IR) 数据模型

IR 是转换后的代码结构化表示，用于代码生成
所有代码转换在 IR 构建之前完成，IR 中存储的是已转换的代码
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WorkflowNodeIR:
    """
    工作流节点 IR

    存储已转换的组件信息，用于代码生成
    """
    name: str                            # 节点名
    class_name: str                      # 生成的类名 (PascalCase + Comp)
    converted_body: str                  # 已转换的函数体代码
    inputs: List[str]                    # 输入字段列表
    outputs: List[str]                   # 输出字段列表
    conversion_source: str               # "rule" 或 "ai"
    docstring: Optional[str] = None      # 文档字符串
    has_llm: bool = False                # 是否使用 LLM
    has_tools: bool = False              # 是否使用工具


@dataclass
class WorkflowEdgeIR:
    """
    工作流边 IR

    存储边的连接信息和条件路由
    """
    source: str                          # 源节点
    target: str                          # 目标节点
    is_conditional: bool = False         # 是否为条件边
    condition_func: Optional[str] = None # 已转换的条件函数代码
    condition_map: Optional[Dict[str, str]] = None  # 条件映射 {"condition": "target_node"}
    router_name: Optional[str] = None    # 路由函数名


@dataclass
class ToolIR:
    """
    工具 IR

    存储工具的转换信息
    """
    name: str                            # 工具名
    func_name: str                       # 函数名
    description: str                     # 描述
    parameters: List[Dict[str, Any]]     # 参数列表
    converted_body: str                  # 已转换的函数体
    return_type: str = "str"             # 返回类型


@dataclass
class LLMConfigIR:
    """
    LLM 配置 IR
    """
    model_name: str = "gpt-4"            # 模型名称
    temperature: float = 0.7             # 温度
    api_base: Optional[str] = None       # API 基础地址
    other_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentIR:
    """
    Agent IR

    完整的 Agent 转换结果
    """
    name: str                            # Agent 名称
    llm_config: Optional[LLMConfigIR] = None  # LLM 配置
    tools: List[ToolIR] = field(default_factory=list)  # 工具列表
    state_fields: List[Dict[str, Any]] = field(default_factory=list)  # 状态字段
    global_vars: List[str] = field(default_factory=list)  # 全局变量
    tool_related_vars: List[str] = field(default_factory=list)  # 工具相关变量
    tool_map_var_name: Optional[str] = None  # 工具映射变量名（从源代码提取）
    imports: List[str] = field(default_factory=list)  # 原始导入语句
    initial_inputs: Dict[str, Any] = field(default_factory=dict)  # 初始输入（从 invoke 提取）
    example_inputs: Dict[str, Any] = field(default_factory=dict)  # 示例输入（从 main 函数提取）


@dataclass
class WorkflowIR:
    """
    工作流 IR

    完整的工作流转换结果
    """
    nodes: List[WorkflowNodeIR] = field(default_factory=list)  # 节点列表
    edges: List[WorkflowEdgeIR] = field(default_factory=list)  # 边列表
    entry_node: Optional[str] = None     # 入口节点
    state_class_name: Optional[str] = None  # 原始状态类名

    def get_node_by_name(self, name: str) -> Optional[WorkflowNodeIR]:
        """根据名称获取节点"""
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def get_conditional_edges(self) -> List[WorkflowEdgeIR]:
        """获取所有条件边"""
        return [e for e in self.edges if e.is_conditional]

    def get_outgoing_edges(self, node_name: str) -> List[WorkflowEdgeIR]:
        """获取节点的所有出边"""
        return [e for e in self.edges if e.source == node_name]

    def get_incoming_edges(self, node_name: str) -> List[WorkflowEdgeIR]:
        """获取节点的所有入边"""
        return [e for e in self.edges if e.target == node_name]


@dataclass
class MigrationIR:
    """
    迁移结果 IR

    包含完整的迁移信息
    """
    agent_ir: AgentIR
    workflow_ir: WorkflowIR
    source_files: List[str] = field(default_factory=list)
    conversion_stats: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于 JSON 序列化）"""
        return {
            "agent": {
                "name": self.agent_ir.name,
                "llm_config": {
                    "model_name": self.agent_ir.llm_config.model_name,
                    "temperature": self.agent_ir.llm_config.temperature,
                } if self.agent_ir.llm_config else None,
                "tools": [
                    {"name": t.name, "description": t.description}
                    for t in self.agent_ir.tools
                ],
            },
            "workflow": {
                "entry_node": self.workflow_ir.entry_node,
                "nodes": [
                    {
                        "name": n.name,
                        "class_name": n.class_name,
                        "inputs": n.inputs,
                        "outputs": n.outputs,
                        "conversion_source": n.conversion_source,
                    }
                    for n in self.workflow_ir.nodes
                ],
                "edges": [
                    {
                        "source": e.source,
                        "target": e.target,
                        "is_conditional": e.is_conditional,
                    }
                    for e in self.workflow_ir.edges
                ],
            },
            "stats": self.conversion_stats,
        }

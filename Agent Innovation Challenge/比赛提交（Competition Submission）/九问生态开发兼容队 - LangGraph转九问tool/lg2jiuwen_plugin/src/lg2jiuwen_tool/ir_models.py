"""
IR Models - Intermediate Representation for migration

Platform-agnostic data structures that represent the parsed LangGraph code
before being transformed into openJiuwen code.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class WorkflowNodeType(Enum):
    """Node types in the workflow"""
    START = "start"
    END = "end"
    FUNCTION = "function"
    CONDITIONAL = "conditional"
    TOOL = "tool"
    LLM = "llm"


@dataclass
class StateAccess:
    """Represents a state access pattern in the code"""
    key: str
    access_type: str  # "read" or "write"
    default_value: Optional[str] = None


@dataclass
class StateFieldInfo:
    """Information about a state field"""
    name: str
    type_hint: str
    has_aggregator: bool = False
    aggregator: Optional[str] = None  # e.g., "operator.add"


@dataclass
class ParamInfo:
    """Function parameter information"""
    name: str
    type_hint: Optional[str] = None
    default_value: Optional[str] = None
    description: Optional[str] = None
    required: bool = True


@dataclass
class NodeFunctionInfo:
    """Information about a node function"""
    name: str
    params: List[ParamInfo]
    body: str
    docstring: Optional[str] = None
    state_accesses: List[StateAccess] = field(default_factory=list)
    return_keys: List[str] = field(default_factory=list)
    is_conditional: bool = False
    condition_returns: List[str] = field(default_factory=list)
    # New fields for better code generation
    body_statements: List[str] = field(default_factory=list)
    return_statement: Optional[str] = None
    # LLM usage flag
    uses_llm: bool = False


@dataclass
class EdgeInfo:
    """Information about a graph edge"""
    source: str
    target: str


@dataclass
class ConditionalEdgeInfo:
    """Information about a conditional edge"""
    source_node: str
    condition_function: str
    mapping: Dict[str, str]  # return_value -> target_node


@dataclass
class ToolInfo:
    """Information about a tool definition"""
    name: str
    description: str
    params: List[ParamInfo]
    return_type: Optional[str] = None
    function_body: str = ""


@dataclass
class LLMConfigIR:
    """LLM configuration IR"""
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    provider: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class WorkflowNodeIR:
    """Workflow node intermediate representation"""
    id: str
    node_type: WorkflowNodeType
    name: str
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowEdgeIR:
    """Workflow edge intermediate representation"""
    source_node: str
    target_node: str
    condition: Optional[str] = None
    branch_id: Optional[str] = None


@dataclass
class WorkflowIR:
    """Complete workflow intermediate representation"""
    name: str
    nodes: List[WorkflowNodeIR] = field(default_factory=list)
    edges: List[WorkflowEdgeIR] = field(default_factory=list)
    entry_node: str = "start"

    def get_upstream_nodes(self, node_id: str) -> List[str]:
        """Get nodes that connect to the given node"""
        return [edge.source_node for edge in self.edges if edge.target_node == node_id]

    def get_downstream_nodes(self, node_id: str) -> List[str]:
        """Get nodes that the given node connects to"""
        return [edge.target_node for edge in self.edges if edge.source_node == node_id]


@dataclass
class ToolIR:
    """Tool intermediate representation"""
    name: str
    description: str
    params: List[ParamInfo]
    return_type: Optional[str] = None
    function_body: str = ""


@dataclass
class AgentIR:
    """Complete agent intermediate representation"""
    name: str
    description: str = ""
    agent_type: str = "workflow"
    llm_config: Optional[LLMConfigIR] = None
    tools: List[ToolIR] = field(default_factory=list)
    workflow: Optional[WorkflowIR] = None
    state_fields: List[StateFieldInfo] = field(default_factory=list)
    source_framework: str = "langgraph"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GlobalVarInfo:
    """Information about a global variable"""
    name: str
    value: str  # The value as source code string
    type_hint: Optional[str] = None


@dataclass
class ParseResult:
    """Result of parsing LangGraph source code"""
    graph_name: str
    state_class_name: Optional[str] = None
    state_fields: List[StateFieldInfo] = field(default_factory=list)
    node_functions: Dict[str, NodeFunctionInfo] = field(default_factory=dict)
    conditional_functions: Dict[str, NodeFunctionInfo] = field(default_factory=dict)
    edges: List[EdgeInfo] = field(default_factory=list)
    conditional_edges: List[ConditionalEdgeInfo] = field(default_factory=list)
    tools: List[ToolInfo] = field(default_factory=list)
    entry_point: Optional[str] = None
    llm_config: Optional[LLMConfigIR] = None
    imports: List[str] = field(default_factory=list)
    import_statements: List[str] = field(default_factory=list)  # Full import statement strings
    global_variables: List[GlobalVarInfo] = field(default_factory=list)
    example_inputs: Dict[str, Any] = field(default_factory=dict)  # Example inputs from if __name__ == "__main__"
    raw_source: str = ""

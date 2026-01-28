"""
select_tool 组件
"""

from openjiuwen.core.component.base import WorkflowComponent
from openjiuwen.core.runtime.base import ComponentExecutable, Input, Output
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.context_engine.base import Context
from ..tools import invoke_tool, tool_map


class SelectToolComp(WorkflowComponent, ComponentExecutable):
    """工具执行节点：根据选择调用对应工具"""

    def __init__(self):
        pass

    async def invoke(
        self,
        inputs: Input,
        runtime: Runtime,
        context: Context
    ) -> Output:
        # 初始化输出变量
        result = None
        # 组件逻辑（转换来源: rule）
        selected_tool = inputs.get('selected_tool')
        tool_input = inputs.get('tool_input', '')
        if not selected_tool or selected_tool not in tool_map:
            return {'result': f'未知工具：{selected_tool}，可用工具：Calculator, Weather'}
        result = invoke_tool(selected_tool, tool_input)
        return {'result': result}
"""
think 组件
"""

from openjiuwen.core.component.base import WorkflowComponent
from openjiuwen.core.runtime.base import ComponentExecutable, Input, Output
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.context_engine.base import Context
from ..config import get_llm, LLM_MODEL_NAME


class ThinkComp(WorkflowComponent, ComponentExecutable):
    """思考节点：分析问题并选择合适的工具"""

    def __init__(self, llm=None):
        if llm:
            self._llm = llm
        else:
            self._llm = get_llm()
        self.model_name = LLM_MODEL_NAME

    async def invoke(
        self,
        inputs: Input,
        runtime: Runtime,
        context: Context
    ) -> Output:
        # 初始化输出变量
        tool_input = None
        selected_tool = None
        thought = None
        loop_count = None
        # 组件逻辑（转换来源: rule）
        content = f'用户问题：{runtime.get_global_state("input")}\n\n可用工具：\n1. Calculator - 用于数学加减乘除计算\n2. Weather - 用于查询城市天气\n\n请分析用户意图，选择合适的工具。\n返回格式（严格遵守）：\n工具：<工具名>\n参数：<工具所需参数>\n思考：<一句话说明理由>\n\n示例1：\n工具：Calculator\n参数：100+200\n思考：用户想计算数学表达式\n\n示例2：\n工具：Weather\n参数：北京 今天\n思考：用户想查询天气'
        messages = [{'role': 'user', 'content': content}]
        response = (await self._llm.ainvoke(model_name=self.model_name, messages=messages)).content
        selected_tool = None
        tool_input = ''
        thought = response
        print('response:', response)
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('工具：') or line.startswith('工具:'):
                selected_tool = line.split('：')[-1].split(':')[-1].strip()
            elif line.startswith('参数：') or line.startswith('参数:'):
                tool_input = line.split('：')[-1].split(':')[-1].strip()
            elif line.startswith('思考：') or line.startswith('思考:'):
                thought = line.split('：')[-1].split(':')[-1].strip()
        loop_count = (runtime.get_global_state("loop_count") or 0) + 1
        # 更新全局状态
        runtime.update_global_state({"loop_count": loop_count})
        return {'thought': thought, 'selected_tool': selected_tool, 'tool_input': tool_input, 'loop_count': loop_count}
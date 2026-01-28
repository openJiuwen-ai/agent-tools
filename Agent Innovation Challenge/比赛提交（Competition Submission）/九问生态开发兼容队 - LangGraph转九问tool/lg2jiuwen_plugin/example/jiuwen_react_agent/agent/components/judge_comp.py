"""
judge 组件
"""

from openjiuwen.core.component.base import WorkflowComponent
from openjiuwen.core.runtime.base import ComponentExecutable, Input, Output
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.context_engine.base import Context
from ..config import get_llm, LLM_MODEL_NAME


class JudgeComp(WorkflowComponent, ComponentExecutable):
    """终止判断节点"""

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
        is_end = None
        reason = None
        # 组件逻辑（转换来源: rule）
        content = f'用户问题：{runtime.get_global_state("input")}\n使用工具：{inputs.get('selected_tool', '')}\n工具结果：{inputs.get('result', '')}\n\n问题：根据工具结果，是否已经能够回答用户的问题？不能回答需要给出原因\n返回格式（严格遵守）：\n结果：True 或 False\n原因：<一句话说明> （可选，只有结果为False时才给出原因）\n\n示例1：\n结果：True\n\n示例2：\n结果：False\n原因：天气查询失败，缺失查询日期 \n\n示例3：\n结果：False\n原因：计算器无法处理该输入，请重新输入\n\n'
        messages = [{'role': 'user', 'content': content}]
        reason = None
        response = (((await self._llm.ainvoke(model_name=self.model_name, messages=messages)).content).strip()).lower()
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('结果：') or line.startswith('结果:'):
                res = line.split('：')[-1].split(':')[-1].strip()
                is_end = res in ('true', 'True', 'yes', '是', '1')
            if is_end:
                break
            if line.startswith('原因：') or line.startswith('原因:'):
                reason = line.split('：')[-1].split(':')[-1].strip()
                print(reason)
        # 更新全局状态
        runtime.update_global_state({"is_end": is_end})
        return {'is_end': is_end, 'reason': reason}
"""
AI 语义理解组件

使用 AI 处理规则无法转换的代码
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from openjiuwen.core.component.base import WorkflowComponent
from openjiuwen.core.runtime.base import ComponentExecutable, Input, Output
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.context_engine.base import Context

from ..workflow.state import (
    PendingItem,
    PendingType,
    ConvertedNode,
    ExtractionResult,
)
from ..rules.base import ConversionResult


class AISemanticComp(WorkflowComponent, ComponentExecutable):
    """
    AI 语义理解组件

    功能：
    1. 只处理 pending_items（规则无法处理的部分）
    2. 为每个 pending_item 调用 AI
    3. 将结果合并回 extraction_result
    """

    def __init__(self, llm=None):
        self._llm = llm

    async def invoke(
        self,
        inputs: Input,
        runtime: Runtime,
        context: Context
    ) -> Output:
        # 从 inputs 获取（通过 transformer 传入）
        extraction_result: ExtractionResult = inputs.get("extraction_result")

        if extraction_result is None:
            raise ValueError("无法获取 extraction_result")

        pending_items = extraction_result.pending_items

        if not pending_items:
            return {"extraction_result": extraction_result}

        # 处理每个 pending_item
        for item in pending_items:
            converted = await self._convert_with_ai(item)

            # 将转换结果添加到 nodes
            extraction_result.nodes.append(ConvertedNode(
                name=self._extract_name(item.id),
                original_code=item.source_code,
                converted_body=converted.code,
                inputs=converted.inputs,
                outputs=converted.outputs,
                conversion_source="ai"
            ))
            extraction_result.ai_count += 1

        # 清空 pending_items
        extraction_result.pending_items = []

        return {"extraction_result": extraction_result}

    def _extract_name(self, item_id: str) -> str:
        """从 item_id 提取名称"""
        # item_id 格式: "file.py:func_name"
        if ":" in item_id:
            return item_id.split(":")[-1]
        return item_id

    async def _convert_with_ai(self, item: PendingItem) -> ConversionResult:
        """调用 AI 转换代码"""
        if self._llm is None:
            # 如果没有 LLM，返回占位符
            return self._fallback_conversion(item)

        system_prompt = self._get_system_prompt(item.pending_type)
        user_prompt = self._build_user_prompt(item)

        try:
            response = await self._llm.ainvoke(
                model_name="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            code = self._extract_code(response.content)
            inputs, outputs = self._analyze_io(code)

            return ConversionResult.success_result(
                code=code,
                inputs=inputs,
                outputs=outputs
            )
        except Exception as e:
            return self._fallback_conversion(item, str(e))

    def _get_system_prompt(self, pending_type: PendingType) -> str:
        """获取系统提示"""
        base = """你是代码转换专家，将 LangGraph 代码转换为 openJiuwen 格式。

openJiuwen 组件规范：
- 异步方法: async def invoke(self, inputs, runtime, context)
- 输入访问: inputs["field_name"]
- 输出返回: return {"field": value}
- LLM调用: await self._llm.ainvoke(model_name=self.model_name, messages=[...])

只输出转换后的代码，不要解释。"""

        if pending_type == PendingType.CONDITIONAL:
            base += """

条件路由函数规范：
- 函数签名: def router(runtime: WorkflowRuntime) -> str
- 状态访问: runtime.get_global_state('node_name.field_name')
- 返回值是目标节点名字符串
- END 转换为 "end"
"""
        return base

    def _build_user_prompt(self, item: PendingItem) -> str:
        """构建用户提示"""
        return f"""
## 上下文
- 状态字段: {item.context.get('state_fields', [])}
- 可用工具: {item.context.get('available_tools', [])}

## 原始代码
```python
{item.source_code}
```

## 问题
{item.question}
"""

    def _extract_code(self, response: str) -> str:
        """从响应中提取代码"""
        # 尝试提取 ```python ... ``` 块
        pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()

        # 尝试提取 ``` ... ``` 块
        pattern = r"```\s*(.*?)\s*```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()

        # 直接返回去掉首尾空白的响应
        return response.strip()

    def _analyze_io(self, code: str) -> Tuple[List[str], List[str]]:
        """分析代码中的输入输出"""
        inputs: List[str] = []
        outputs: List[str] = []

        # 查找 inputs["xxx"]
        input_pattern = r'inputs\["(\w+)"\]|inputs\.get\("(\w+)"'
        for match in re.finditer(input_pattern, code):
            key = match.group(1) or match.group(2)
            if key and key not in inputs:
                inputs.append(key)

        # 查找 return {"xxx": ...}
        # 简单解析，查找 "key":
        output_pattern = r'"(\w+)":'
        for match in re.finditer(output_pattern, code):
            key = match.group(1)
            if key and key not in outputs:
                outputs.append(key)

        return inputs, outputs

    def _fallback_conversion(
        self,
        item: PendingItem,
        error: Optional[str] = None
    ) -> ConversionResult:
        """
        回退转换

        当 AI 不可用时，生成带 TODO 注释的代码
        """
        error_comment = f"# AI 转换失败: {error}\n" if error else ""
        code = f"""{error_comment}# TODO: 需要手动转换以下代码
# 原始代码:
# {item.source_code.replace(chr(10), chr(10) + '# ')}
pass"""

        return ConversionResult.success_result(
            code=code,
            inputs=[],
            outputs=[]
        )

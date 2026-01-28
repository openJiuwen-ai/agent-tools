"""
LLM 调用规则

处理 llm.invoke() 和相关调用的转换
"""

import ast
from typing import Optional, Set

from .base import BaseRule, StatementRule, ConversionResult, transform_state_to_inputs


class LLMInvokeRule(StatementRule):
    """
    LLM 调用规则

    转换:
    - response = llm.invoke(messages) → response = await self._llm.ainvoke(model_name=self.model_name, messages=messages)
    - response = model.invoke(messages) → response = await self._llm.ainvoke(...)
    - content = llm.invoke(messages).content → content = (await self._llm.ainvoke(...)).content
    """

    # 已知的 LLM 变量名
    KNOWN_LLM_VARS = {"llm", "model", "chat", "chat_model", "chatmodel"}

    # 已知的 LLM 方法
    KNOWN_LLM_METHODS = {"invoke", "ainvoke", "call", "generate"}

    def matches(self, node: ast.AST) -> bool:
        """判断是否为 LLM 调用"""
        # 匹配赋值语句: response = llm.invoke(...)
        if isinstance(node, ast.Assign):
            return self._is_llm_call(node.value)

        # 匹配表达式语句: llm.invoke(...)
        if isinstance(node, ast.Expr):
            return self._is_llm_call(node.value)

        return False

    def _is_llm_call(self, node: ast.AST) -> bool:
        """检查是否为 LLM 调用"""
        # 直接调用: llm.invoke(...)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                var_name = self._get_var_name(node.func.value)
                method = node.func.attr
                if var_name.lower() in self.KNOWN_LLM_VARS and method in self.KNOWN_LLM_METHODS:
                    return True
            # 方法链调用: llm.invoke(...).content.strip()
            # 检查 func.value 是否包含 LLM 调用
            if isinstance(node.func, ast.Attribute):
                return self._is_llm_call(node.func.value)

        # 属性访问: llm.invoke(...).content
        if isinstance(node, ast.Attribute):
            return self._is_llm_call(node.value)

        return False

    def convert(self, node: ast.AST) -> ConversionResult:
        """转换 LLM 调用"""
        # 先转换 state 访问为 inputs 访问
        transformed_node, inputs_used, outputs_used = transform_state_to_inputs(node)

        if isinstance(transformed_node, ast.Assign):
            target = ast.unparse(transformed_node.targets[0])
            call_code = self._convert_llm_call(transformed_node.value)
            return ConversionResult.success_result(
                code=f'{target} = {call_code}',
                inputs=inputs_used
            )

        if isinstance(transformed_node, ast.Expr):
            call_code = self._convert_llm_call(transformed_node.value)
            return ConversionResult.success_result(code=call_code, inputs=inputs_used)

        return ConversionResult.failure()

    def _convert_llm_call(self, node: ast.AST) -> str:
        """转换 LLM 调用表达式"""
        # 处理属性访问: llm.invoke(...).content
        if isinstance(node, ast.Attribute):
            inner = self._convert_llm_call(node.value)
            if "await" in inner:
                # 已经是 await 表达式，需要包装
                return f'({inner}).{node.attr}'
            return f'{inner}.{node.attr}'

        # 处理方法调用: .strip() 等
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                var_name = self._get_var_name(node.func.value)
                method = node.func.attr

                # 检查是否为 LLM 调用
                if var_name.lower() in self.KNOWN_LLM_VARS and method in self.KNOWN_LLM_METHODS:
                    # 这是 LLM 调用
                    messages_arg = "messages"
                    if node.args:
                        messages_arg = ast.unparse(node.args[0])
                    elif node.keywords:
                        for kw in node.keywords:
                            if kw.arg == "messages":
                                messages_arg = ast.unparse(kw.value)
                                break
                    return f'await self._llm.ainvoke(model_name=self.model_name, messages={messages_arg})'
                else:
                    # 这是链式调用如 .strip()
                    inner = self._convert_llm_call(node.func.value)
                    # 构建参数
                    args_str = ", ".join(ast.unparse(a) for a in node.args)
                    kwargs_str = ", ".join(f"{kw.arg}={ast.unparse(kw.value)}" for kw in node.keywords if kw.arg)
                    all_args = ", ".join(filter(None, [args_str, kwargs_str]))
                    if "await" in inner:
                        return f'({inner}).{method}({all_args})'
                    return f'{inner}.{method}({all_args})'

        return ast.unparse(node)


class LLMBindRule(StatementRule):
    """
    LLM bind 规则

    转换:
    - bound_llm = llm.bind_tools([...]) → 记录工具绑定
    """

    def matches(self, node: ast.AST) -> bool:
        """判断是否为 bind_tools 调用"""
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Attribute):
                    return node.value.func.attr in ("bind_tools", "bind")
        return False

    def convert(self, node: ast.AST) -> ConversionResult:
        """转换 bind_tools"""
        # 在 openJiuwen 中，工具绑定方式不同
        # 这里生成注释提示
        if isinstance(node, ast.Assign):
            target = ast.unparse(node.targets[0])
            original = ast.unparse(node.value)
            return ConversionResult.success_result(
                code=f'# TODO: 工具绑定需要在组件初始化时配置\n        # 原始代码: {target} = {original}\n        {target} = self._llm'
            )

        return ConversionResult.failure()

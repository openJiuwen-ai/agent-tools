"""
规则基类定义

所有转换规则都继承自 BaseRule
"""

import ast
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


class StateToInputsTransformer(ast.NodeTransformer):
    """
    AST 转换器：
    - 将 state["x"] 和 state.get("x") 递归转换为 inputs["x"] 和 inputs.get("x")（读取）
    - 将 state["x"] = value 转换为 x = value（写入）
    """

    def __init__(self):
        self.inputs_used: List[str] = []
        self.outputs_used: List[str] = []

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        """处理赋值语句，转换 state["x"] = value -> x = value"""
        # 先递归处理右侧值
        node.value = self.visit(node.value)

        new_targets = []
        for target in node.targets:
            if isinstance(target, ast.Subscript):
                if isinstance(target.value, ast.Name) and target.value.id in ("state", "State"):
                    # state["key"] = value -> key = value
                    key = self._extract_string_key(target.slice)
                    if key:
                        self.outputs_used.append(key)
                        # 替换为简单变量赋值
                        new_targets.append(ast.Name(id=key, ctx=ast.Store()))
                        continue
            # 递归处理其他目标
            new_targets.append(self.visit(target))

        node.targets = new_targets
        return node

    def visit_Return(self, node: ast.Return) -> ast.AST:
        """处理 return 语句，转换 return state -> return __COLLECTED_OUTPUTS__"""
        if node.value is not None:
            # return state
            if isinstance(node.value, ast.Name) and node.value.id in ("state", "State"):
                # 替换为占位符，让代码生成器处理
                node.value = ast.Name(id="__COLLECTED_OUTPUTS__", ctx=ast.Load())
            else:
                # 递归处理返回值中的 state 访问
                node.value = self.visit(node.value)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        """转换 state["x"] -> inputs["x"]（读取）"""
        # 先递归处理子节点
        self.generic_visit(node)

        # 只处理 Load 上下文（读取），Store 上下文在 visit_Assign 中处理
        if isinstance(node.ctx, ast.Load):
            if isinstance(node.value, ast.Name) and node.value.id in ("state", "State"):
                # 提取 key
                key = self._extract_string_key(node.slice)
                if key:
                    self.inputs_used.append(key)
                    # 替换 state -> inputs
                    node.value = ast.Name(id="inputs", ctx=ast.Load())
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """转换 state.get("x") -> inputs.get("x")"""
        # 先递归处理子节点
        self.generic_visit(node)

        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "get":
                if isinstance(node.func.value, ast.Name) and node.func.value.id in ("state", "State"):
                    if node.args:
                        key = self._extract_string_key(node.args[0])
                        if key:
                            self.inputs_used.append(key)
                            # 替换 state -> inputs
                            node.func.value = ast.Name(id="inputs", ctx=ast.Load())
        return node

    def _extract_string_key(self, node: ast.AST) -> Optional[str]:
        """从 AST 节点提取字符串键"""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Str):  # Python 3.7 兼容
            return node.s
        return None


def transform_state_to_inputs(node: ast.AST) -> Tuple[ast.AST, List[str], List[str]]:
    """
    转换 AST 节点中的 state 访问

    - state["x"] 读取 -> inputs["x"]
    - state["x"] = value 写入 -> x = value

    Returns:
        (转换后的节点, 使用的 inputs 列表, 产生的 outputs 列表)
    """
    # 深拷贝避免修改原始 AST
    node_copy = copy.deepcopy(node)
    transformer = StateToInputsTransformer()
    transformed = transformer.visit(node_copy)
    ast.fix_missing_locations(transformed)
    return transformed, transformer.inputs_used, transformer.outputs_used


@dataclass
class ConversionResult:
    """
    转换结果

    规则转换后返回此结果
    """
    success: bool = True                 # 是否转换成功
    code: str = ""                       # 转换后的代码
    inputs: List[str] = field(default_factory=list)   # 使用的输入字段
    outputs: List[str] = field(default_factory=list)  # 产生的输出字段
    failed_lines: List[Dict[str, Any]] = field(default_factory=list)  # 失败的行
    error_message: Optional[str] = None  # 错误信息

    @staticmethod
    def failure(
        failed_lines: Optional[List[Dict[str, Any]]] = None,
        error_message: Optional[str] = None
    ) -> "ConversionResult":
        """创建失败结果"""
        return ConversionResult(
            success=False,
            failed_lines=failed_lines or [],
            error_message=error_message
        )

    @staticmethod
    def success_result(
        code: str,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None
    ) -> "ConversionResult":
        """创建成功结果"""
        return ConversionResult(
            success=True,
            code=code,
            inputs=inputs or [],
            outputs=outputs or []
        )


class BaseRule(ABC):
    """
    规则基类

    所有转换规则必须继承此类并实现 matches 和 convert 方法
    """

    @abstractmethod
    def matches(self, node: ast.AST) -> bool:
        """
        判断是否匹配此规则

        Args:
            node: AST 节点

        Returns:
            是否匹配
        """
        pass

    @abstractmethod
    def convert(self, node: ast.AST) -> ConversionResult:
        """
        执行转换

        Args:
            node: AST 节点

        Returns:
            转换结果
        """
        pass

    def _is_state_var(self, node: ast.AST) -> bool:
        """判断是否为状态变量"""
        if isinstance(node, ast.Name):
            return node.id in ("state", "State")
        return False

    def _get_var_name(self, node: ast.AST) -> str:
        """获取变量名"""
        if isinstance(node, ast.Name):
            return node.id
        return ""

    def _extract_string_key(self, node: ast.AST) -> Optional[str]:
        """从 AST 节点提取字符串键"""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Str):  # Python 3.7 兼容
            return node.s
        return None


class StatementRule(BaseRule):
    """
    语句级规则基类

    用于处理完整的语句（如赋值、返回等）
    """

    def matches_statement(self, stmt: ast.stmt) -> bool:
        """判断是否匹配语句"""
        return self.matches(stmt)

    def convert_statement(self, stmt: ast.stmt) -> ConversionResult:
        """转换语句"""
        return self.convert(stmt)


class ExpressionRule(BaseRule):
    """
    表达式级规则基类

    用于处理表达式（如函数调用、属性访问等）
    """

    def matches_expr(self, expr: ast.expr) -> bool:
        """判断是否匹配表达式"""
        return self.matches(expr)

    def convert_expr(self, expr: ast.expr) -> ConversionResult:
        """转换表达式"""
        return self.convert(expr)


class PassthroughRule(BaseRule):
    """
    透传规则

    对于不需要特殊处理的语句，保持原样，但会递归转换其中的 state 访问
    """

    # 不需要转换的语句类型
    PASSTHROUGH_TYPES = (
        ast.Import,       # import xxx
        ast.ImportFrom,   # from xxx import yyy
        ast.Expr,         # 表达式语句（如 print）
        ast.If,           # if 语句
        ast.For,          # for 循环
        ast.While,        # while 循环
        ast.Try,          # try/except
        ast.With,         # with 语句
        ast.FunctionDef,  # 内部函数定义
        ast.ClassDef,     # 内部类定义
    )

    def matches(self, node: ast.AST) -> bool:
        """匹配不需要特殊处理的语句"""
        # 简单赋值（不是 state 赋值）
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if isinstance(target, ast.Name):
                # 普通变量赋值，需要转换其中的 state 访问
                return True
            if isinstance(target, ast.Subscript):
                # 只有非 state 的下标赋值才透传
                if isinstance(target.value, ast.Name):
                    if target.value.id not in ("state", "State"):
                        return True
        # 其他透传类型
        return isinstance(node, self.PASSTHROUGH_TYPES)

    def convert(self, node: ast.AST) -> ConversionResult:
        """转换 state 访问后保持原样"""
        # 递归转换 state["x"] -> inputs["x"]，state["x"] = v -> x = v
        transformed, inputs_used, outputs_used = transform_state_to_inputs(node)
        code = ast.unparse(transformed)
        return ConversionResult.success_result(code=code, inputs=inputs_used, outputs=outputs_used)


class RuleChain:
    """
    规则链

    按顺序尝试应用多个规则
    """

    def __init__(self, rules: Optional[List[BaseRule]] = None):
        self.rules: List[BaseRule] = rules or []

    def add_rule(self, rule: BaseRule) -> "RuleChain":
        """添加规则"""
        self.rules.append(rule)
        return self

    def try_convert(self, node: ast.AST) -> ConversionResult:
        """
        尝试用规则链转换节点

        按顺序尝试每个规则，返回第一个匹配的结果
        """
        for rule in self.rules:
            if rule.matches(node):
                return rule.convert(node)
        return ConversionResult.failure(error_message="No matching rule")

    def convert_statements(
        self,
        statements: List[ast.stmt]
    ) -> ConversionResult:
        """
        转换语句列表

        返回合并的转换结果
        """
        converted_lines: List[str] = []
        all_inputs: Set[str] = set()
        all_outputs: Set[str] = set()
        failed_lines: List[Dict[str, Any]] = []

        for stmt in statements:
            # 跳过 docstring
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                if isinstance(stmt.value.value, str):
                    continue

            result = self.try_convert(stmt)

            if result.success:
                converted_lines.append(result.code)
                all_inputs.update(result.inputs)
                all_outputs.update(result.outputs)
            else:
                failed_lines.append({
                    "line": getattr(stmt, "lineno", 0),
                    "code": ast.unparse(stmt)
                })

        if failed_lines:
            return ConversionResult.failure(failed_lines=failed_lines)

        return ConversionResult.success_result(
            code="\n".join(converted_lines),
            inputs=list(all_inputs),
            outputs=list(all_outputs)
        )

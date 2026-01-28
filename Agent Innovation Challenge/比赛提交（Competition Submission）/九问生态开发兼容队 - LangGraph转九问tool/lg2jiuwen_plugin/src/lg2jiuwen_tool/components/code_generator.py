"""
代码生成组件

根据 IR 生成 openJiuwen 代码
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from openjiuwen.core.component.base import WorkflowComponent
from openjiuwen.core.runtime.base import ComponentExecutable, Input, Output
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.runtime.workflow import WorkflowRuntime
from openjiuwen.core.context_engine.base import Context

from ..ir.models import (
    AgentIR,
    WorkflowIR,
    WorkflowNodeIR,
    WorkflowEdgeIR,
    ToolIR,
    MigrationIR,
)


class CodeGeneratorComp(WorkflowComponent, ComponentExecutable):
    """
    代码生成组件

    功能：
    - 根据 IR 生成 openJiuwen 代码
    - 纯模板填充，不需要 AI
    - IR 中已包含转换后的代码
    """

    async def invoke(
        self,
        inputs: Input,
        runtime: Runtime,
        context: Context
    ) -> Output:
        # 从 inputs 获取（通过 transformer 传入）
        agent_ir: AgentIR = inputs.get("agent_ir")
        workflow_ir: WorkflowIR = inputs.get("workflow_ir")
        migration_ir: MigrationIR = inputs.get("migration_ir")
        output_dir: str = inputs.get("output_dir", "./output")
        is_multi_file: bool = inputs.get("is_multi_file", False)
        project_root: str = inputs.get("project_root", "")

        if agent_ir is None or workflow_ir is None:
            raise ValueError("无法获取 agent_ir 或 workflow_ir")

        os.makedirs(output_dir, exist_ok=True)
        generated_files = []

        if is_multi_file:
            # 多文件模式：生成与源目录类似的结构
            generated_files = self._gen_multi_file_output(
                agent_ir, workflow_ir, migration_ir, output_dir, project_root
            )
            generated_code = "# 多文件项目，请查看生成的目录结构"
        else:
            # 单文件模式：生成单个文件
            sections = []

            # 1. 生成导入语句
            sections.append(self._gen_imports(agent_ir, workflow_ir))

            # 2. 生成工具函数
            for tool in agent_ir.tools:
                sections.append(self._gen_tool(tool))

            # 2.5 生成工具映射和 invoke_tool
            if agent_ir.tools:
                sections.append(self._gen_tool_map_and_invoke(agent_ir))

            # 3. 生成组件类
            for node in workflow_ir.nodes:
                sections.append(self._gen_component(node, agent_ir))

            # 4. 生成路由函数
            for edge in workflow_ir.edges:
                if edge.is_conditional and edge.condition_func:
                    sections.append(edge.condition_func)

            # 5. 生成工作流构建函数
            sections.append(self._gen_workflow_builder(workflow_ir, agent_ir))

            # 6. 生成主函数
            sections.append(self._gen_main(agent_ir))

            # 合并代码
            generated_code = "\n\n\n".join(sections)

            # 写入生成的代码
            output_file = os.path.join(output_dir, f"{agent_ir.name.lower()}_openjiuwen.py")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(generated_code)
            generated_files.append(output_file)

        # 生成 IR 结果文件
        ir_file = os.path.join(output_dir, f"{agent_ir.name.lower()}_ir.json")
        ir_data = self._serialize_ir(agent_ir, workflow_ir, migration_ir)
        with open(ir_file, "w", encoding="utf-8") as f:
            json.dump(ir_data, f, ensure_ascii=False, indent=2)
        generated_files.append(ir_file)

        # 生成迁移报告
        report = self._gen_report(agent_ir, workflow_ir, migration_ir, generated_files)
        report_file = os.path.join(output_dir, f"{agent_ir.name.lower()}_report.md")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        generated_files.append(report_file)

        return {
            "generated_code": generated_code,
            "generated_files": generated_files,
            "output_dir": output_dir,
            "report": report
        }

    def _gen_multi_file_output(
        self,
        agent_ir: AgentIR,
        workflow_ir: WorkflowIR,
        migration_ir: Optional[MigrationIR],
        output_dir: str,
        project_root: str
    ) -> List[str]:
        """
        生成多文件输出结构

        结构:
        {agent_name}/
        ├── __init__.py
        ├── config.py          # 配置（LLM配置、全局变量）
        ├── tools.py           # 工具函数
        ├── components/
        │   ├── __init__.py
        │   └── {node}_comp.py # 每个节点一个组件文件
        ├── routers.py         # 路由函数
        ├── workflow.py        # 工作流构建
        └── main.py            # 主入口
        """
        generated_files = []
        agent_name = agent_ir.name.lower()

        # 创建目录结构
        agent_dir = os.path.join(output_dir, agent_name)
        components_dir = os.path.join(agent_dir, "components")
        os.makedirs(components_dir, exist_ok=True)

        # 1. 生成 __init__.py
        init_file = os.path.join(agent_dir, "__init__.py")
        init_content = f'''"""
{agent_ir.name} - 由 LG2Jiuwen 自动迁移生成
"""

from .workflow import build_{agent_name}_workflow

__all__ = ["build_{agent_name}_workflow"]
'''
        with open(init_file, "w", encoding="utf-8") as f:
            f.write(init_content)
        generated_files.append(init_file)

        # 2. 生成 config.py
        config_file = os.path.join(agent_dir, "config.py")
        config_content = self._gen_config_file(agent_ir)
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(config_content)
        generated_files.append(config_file)

        # 3. 生成 tools.py
        if agent_ir.tools:
            tools_file = os.path.join(agent_dir, "tools.py")
            tools_content = self._gen_tools_file(agent_ir)
            with open(tools_file, "w", encoding="utf-8") as f:
                f.write(tools_content)
            generated_files.append(tools_file)

        # 4. 生成 components/__init__.py
        comp_init_file = os.path.join(components_dir, "__init__.py")
        comp_imports = []
        for node in workflow_ir.nodes:
            comp_imports.append(f"from .{node.name}_comp import {node.class_name}")
        comp_init_content = f'''"""
组件模块
"""

{chr(10).join(comp_imports)}

__all__ = [{", ".join(f'"{n.class_name}"' for n in workflow_ir.nodes)}]
'''
        with open(comp_init_file, "w", encoding="utf-8") as f:
            f.write(comp_init_content)
        generated_files.append(comp_init_file)

        # 5. 生成各个组件文件
        for node in workflow_ir.nodes:
            comp_file = os.path.join(components_dir, f"{node.name}_comp.py")
            comp_content = self._gen_component_file(node, agent_ir)
            with open(comp_file, "w", encoding="utf-8") as f:
                f.write(comp_content)
            generated_files.append(comp_file)

        # 6. 生成 routers.py
        routers_file = os.path.join(agent_dir, "routers.py")
        routers_content = self._gen_routers_file(workflow_ir, agent_ir)
        with open(routers_file, "w", encoding="utf-8") as f:
            f.write(routers_content)
        generated_files.append(routers_file)

        # 7. 生成 workflow.py
        workflow_file = os.path.join(agent_dir, "workflow.py")
        workflow_content = self._gen_workflow_file(workflow_ir, agent_ir)
        with open(workflow_file, "w", encoding="utf-8") as f:
            f.write(workflow_content)
        generated_files.append(workflow_file)

        # 8. 生成 main.py
        main_file = os.path.join(agent_dir, "main.py")
        main_content = self._gen_main_file(agent_ir, workflow_ir)
        with open(main_file, "w", encoding="utf-8") as f:
            f.write(main_content)
        generated_files.append(main_file)

        return generated_files

    def _gen_config_file(self, agent_ir: AgentIR) -> str:
        """生成配置文件"""
        lines = [
            '"""',
            '配置文件',
            '"""',
            '',
            'import os',
        ]

        # 添加 LLM 相关导入
        if agent_ir.llm_config:
            lines.append('from openjiuwen.core.utils.llm.model_library.openai import OpenAIChatModel')

        lines.append('')
        lines.append("# SSL 配置")
        lines.append("os.environ['LLM_SSL_VERIFY'] = 'false'")
        lines.append('')

        # 分类全局变量：LLM 相关 vs 其他
        llm_vars = []
        other_vars = []
        llm_keywords = ['LLM', 'MODEL', 'API_KEY', 'API_BASE', 'TEMPERATURE']

        for var in agent_ir.global_vars:
            # 提取变量名（等号左边）
            var_name = var.split('=')[0].strip() if '=' in var else ''
            is_llm_related = any(kw in var_name.upper() for kw in llm_keywords)
            if is_llm_related:
                llm_vars.append(var)
            else:
                other_vars.append(var)

        # 添加非 LLM 全局变量
        if other_vars:
            lines.append('# 全局变量')
            for var in other_vars:
                lines.append(var)
            lines.append('')

        # 添加 LLM 配置（保持源文件变量名）
        if llm_vars:
            lines.append('# LLM 配置')
            for var in llm_vars:
                lines.append(var)
            lines.append('')

        # 生成 get_llm 函数，使用源文件的变量名
        if agent_ir.llm_config or llm_vars:
            # 从 llm_vars 中查找 API_KEY 和 API_BASE 变量名
            api_key_var = self._find_var_name(llm_vars, ['API_KEY', 'LLM_API_KEY', 'OPENAI_API_KEY'])
            api_base_var = self._find_var_name(llm_vars, ['API_BASE', 'LLM_API_BASE', 'OPENAI_API_BASE'])

            if api_key_var and api_base_var:
                lines.extend([
                    'def get_llm():',
                    '    """获取 LLM 实例"""',
                    f'    return OpenAIChatModel(api_key={api_key_var}, api_base={api_base_var})',
                ])

        return '\n'.join(lines)

    def _find_var_name(self, var_list: List[str], candidates: List[str]) -> Optional[str]:
        """从变量列表中查找匹配的变量名"""
        for var in var_list:
            var_name = var.split('=')[0].strip() if '=' in var else ''
            for candidate in candidates:
                if var_name.upper() == candidate.upper():
                    return var_name
        # 如果找不到，返回第一个候选项
        return candidates[0] if candidates else None

    def _find_model_name_var(self, agent_ir: AgentIR) -> str:
        """从全局变量中查找模型名称变量"""
        # 候选变量名（按优先级排列）
        candidates = ['MODEL_NAME', 'LLM_MODEL_NAME', 'LLM_MODEL', 'MODEL']

        for var in agent_ir.global_vars:
            var_name = var.split('=')[0].strip() if '=' in var else ''
            var_upper = var_name.upper()
            for candidate in candidates:
                if var_upper == candidate:
                    return var_name

        # 如果找不到精确匹配，尝试模糊匹配（包含 MODEL 且包含 NAME）
        for var in agent_ir.global_vars:
            var_name = var.split('=')[0].strip() if '=' in var else ''
            var_upper = var_name.upper()
            if 'MODEL' in var_upper and 'NAME' in var_upper:
                return var_name

        # 如果还是找不到，尝试只包含 MODEL 的变量
        for var in agent_ir.global_vars:
            var_name = var.split('=')[0].strip() if '=' in var else ''
            var_upper = var_name.upper()
            if 'MODEL' in var_upper and 'API' not in var_upper and 'BASE' not in var_upper:
                return var_name

        # 默认返回 MODEL_NAME
        return 'MODEL_NAME'

    def _gen_tools_file(self, agent_ir: AgentIR) -> str:
        """生成工具文件"""
        lines = [
            '"""',
            '工具函数',
            '"""',
            '',
        ]

        # 检查需要的导入
        tool_code = "\n".join(t.converted_body for t in agent_ir.tools)
        if "httpx" in tool_code:
            lines.append('import httpx')

        lines.extend([
            'from openjiuwen.core.utils.tool.param import Param',
            'from openjiuwen.core.utils.tool.tool import tool',
            '',
            'from .config import *',
            '',
        ])

        # 添加工具函数
        for tool in agent_ir.tools:
            lines.append(self._gen_tool(tool))
            lines.append('')

        # 添加工具映射变量
        tool_map_name = agent_ir.tool_map_var_name or "tool_map"
        if agent_ir.tool_related_vars:
            # 使用从源代码提取的工具映射
            lines.append('# 工具映射')
            for var in agent_ir.tool_related_vars:
                lines.append(var)
            lines.append('')
        elif agent_ir.tools:
            # 自动生成工具映射
            lines.append('# 工具映射')
            tool_entries = ', '.join(
                f'"{t.name}": {t.func_name}' for t in agent_ir.tools
            )
            lines.append(f'{tool_map_name} = {{{tool_entries}}}')
            lines.append('')

        # 添加 invoke_tool 辅助函数
        lines.extend([
            '',
            'def invoke_tool(tool_name: str, arg: str) -> str:',
            '    """',
            '    调用工具的辅助函数',
            '',
            '    openJiuwen 的 @tool 装饰器返回 LocalFunction，',
            '    需要通过 .invoke(inputs={param_name: arg}) 调用。',
            '    此函数自动处理参数名映射。',
            '    """',
            f'    tool_func = {tool_map_name}.get(tool_name)',
            '    if tool_func is None:',
            '        return f"未知工具: {tool_name}"',
            '    # 获取工具的第一个参数名',
            '    if hasattr(tool_func, "params") and tool_func.params:',
            '        param_name = tool_func.params[0].name',
            '    else:',
            '        param_name = "input"',
            '    return tool_func.invoke(inputs={param_name: arg})',
            '',
        ])

        return '\n'.join(lines)

    def _gen_component_file(self, node: WorkflowNodeIR, agent_ir: AgentIR) -> str:
        """生成单个组件文件"""
        lines = [
            '"""',
            f'{node.name} 组件',
            '"""',
            '',
        ]

        # 检查是否需要 json 模块
        if 'json.loads' in node.converted_body or 'json.dumps' in node.converted_body:
            lines.append('import json')
            lines.append('')

        lines.extend([
            'from openjiuwen.core.component.base import WorkflowComponent',
            'from openjiuwen.core.runtime.base import ComponentExecutable, Input, Output',
            'from openjiuwen.core.runtime.runtime import Runtime',
            'from openjiuwen.core.context_engine.base import Context',
        ])

        # 添加 config 导入
        config_imports = []
        if node.has_llm:
            model_name_var = self._find_model_name_var(agent_ir)
            config_imports.extend(['get_llm', model_name_var])
        if config_imports:
            lines.append(f'from ..config import {", ".join(config_imports)}')

        # 添加工具导入 - 检查代码中是否使用了工具映射变量、invoke_tool 或工具函数
        imports_list = []
        # 检查是否使用了 invoke_tool（优先检查）
        if 'invoke_tool(' in node.converted_body:
            imports_list.append('invoke_tool')
        # 检查是否使用了工具映射变量（使用从源代码提取的变量名）
        tool_map_name = agent_ir.tool_map_var_name or "tool_map"
        if tool_map_name in node.converted_body:
            imports_list.append(tool_map_name)
        # 检查是否使用了单个工具函数
        if agent_ir.tools:
            for tool in agent_ir.tools:
                func_name = tool.func_name or tool.name.lower()
                # 检查代码中是否直接调用了该工具函数
                if func_name + '(' in node.converted_body or func_name + ' ' in node.converted_body:
                    if func_name not in imports_list:
                        imports_list.append(func_name)
        if imports_list:
            lines.append(f'from ..tools import {", ".join(imports_list)}')

        lines.append('')
        lines.append('')

        # 生成组件类（多文件模式）
        lines.append(self._gen_component_multi_file(node, agent_ir))

        return '\n'.join(lines)

    def _gen_component_multi_file(self, node: WorkflowNodeIR, agent_ir: AgentIR) -> str:
        """生成组件类（多文件模式，使用 get_llm）

        数据传递方式：
        1. return 的值自动同步给下游，存储为 {节点名}.{字段名}
        2. 全局状态变量需显式调用 runtime.update_global_state() 更新，
           所有组件都能通过 runtime.get_global_state("key") 访问
        """
        # 生成初始化方法
        init_method = self._gen_init_method_multi_file(node, agent_ir)

        # 生成输出初始化
        output_init = self._gen_output_init(node.outputs)

        # 生成输出字典（所有输出通过 return 传递给下游）
        outputs_dict = "{" + ", ".join(f'"{o}": {o}' for o in node.outputs) + "}" if node.outputs else "{}"

        # 确定哪些是全局状态变量（初始输入）
        global_state_keys = set(agent_ir.initial_inputs.keys()) if agent_ir.initial_inputs else set()
        global_outputs = [o for o in node.outputs if o in global_state_keys]

        # 处理组件逻辑代码
        body_code = node.converted_body
        body_code = body_code.replace("__COLLECTED_OUTPUTS__", outputs_dict)

        # 将全局状态变量的访问从 inputs.get/inputs[] 转换为 runtime.get_global_state()
        body_code = self._convert_global_state_access(body_code, global_state_keys)

        docstring = node.docstring or f"{node.name} 组件"

        # 检查代码是否已经以 return 结尾
        body_lines = [l for l in body_code.strip().split('\n') if l.strip()]
        has_final_return = body_lines and body_lines[-1].strip().startswith('return ')

        # 生成更新全局状态的代码
        if global_outputs:
            global_dict = "{" + ", ".join(f'"{o}": {o}' for o in global_outputs) + "}"
            update_global_state = f"runtime.update_global_state({global_dict})"
        else:
            update_global_state = ""

        if has_final_return:
            # 在 return 之前插入 update_global_state
            if update_global_state:
                body_code = self._insert_global_state_update(body_code, global_outputs)
            return f'''class {node.class_name}(WorkflowComponent, ComponentExecutable):
    """{docstring}"""

{init_method}

    async def invoke(
        self,
        inputs: Input,
        runtime: Runtime,
        context: Context
    ) -> Output:
{output_init}
        # 组件逻辑（转换来源: {node.conversion_source}）
{self._indent(body_code, 8)}'''
        else:
            ending_code = ""
            if update_global_state:
                ending_code = f"\n        # 更新全局状态\n        {update_global_state}"

            return f'''class {node.class_name}(WorkflowComponent, ComponentExecutable):
    """{docstring}"""

{init_method}

    async def invoke(
        self,
        inputs: Input,
        runtime: Runtime,
        context: Context
    ) -> Output:
{output_init}
        # 组件逻辑（转换来源: {node.conversion_source}）
{self._indent(body_code, 8)}{ending_code}
        return {outputs_dict}'''

    def _convert_global_state_access(self, body_code: str, global_state_keys: set) -> str:
        """
        将全局状态变量的访问从 inputs 转换为 runtime.get_global_state()

        转换规则：
        - inputs.get("key", default) → runtime.get_global_state("key") or default
        - inputs.get("key") → runtime.get_global_state("key")
        - inputs["key"] → runtime.get_global_state("key")

        只对 global_state_keys 中的变量进行转换
        """
        import re

        if not global_state_keys:
            return body_code

        result = body_code

        for key in global_state_keys:
            # 转换 inputs.get("key", default) → (runtime.get_global_state("key") or default)
            # 注意：需要处理默认值
            pattern_with_default = rf'inputs\.get\(["\']({re.escape(key)})["\']\s*,\s*([^)]+)\)'
            result = re.sub(
                pattern_with_default,
                lambda m: f'(runtime.get_global_state("{m.group(1)}") or {m.group(2).strip()})',
                result
            )

            # 转换 inputs.get("key") → runtime.get_global_state("key")
            pattern_no_default = rf'inputs\.get\(["\']({re.escape(key)})["\']\)'
            result = re.sub(
                pattern_no_default,
                lambda m: f'runtime.get_global_state("{m.group(1)}")',
                result
            )

            # 转换 inputs["key"] → runtime.get_global_state("key")
            pattern_subscript = rf'inputs\[["\']({re.escape(key)})["\']\]'
            result = re.sub(
                pattern_subscript,
                lambda m: f'runtime.get_global_state("{m.group(1)}")',
                result
            )

        return result

    def _insert_global_state_update(self, body_code: str, global_outputs: List[str]) -> str:
        """在 return 语句之前插入全局状态更新"""
        if not global_outputs:
            return body_code

        global_dict = "{" + ", ".join(f'"{o}": {o}' for o in global_outputs) + "}"

        lines = body_code.split('\n')
        result_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('return '):
                # 获取当前行的缩进
                indent = len(line) - len(line.lstrip())
                indent_str = ' ' * indent
                # 在 return 之前插入更新全局状态
                result_lines.append(f'{indent_str}# 更新全局状态')
                result_lines.append(f'{indent_str}runtime.update_global_state({global_dict})')
            result_lines.append(line)

        return '\n'.join(result_lines)

    def _gen_init_method_multi_file(self, node: WorkflowNodeIR, agent_ir: AgentIR) -> str:
        """生成初始化方法（多文件模式，使用 get_llm）"""
        if node.has_llm:
            model_name_var = self._find_model_name_var(agent_ir)
            return f'''    def __init__(self, llm=None):
        if llm:
            self._llm = llm
        else:
            self._llm = get_llm()
        self.model_name = {model_name_var}'''
        else:
            return '''    def __init__(self):
        pass'''

    def _gen_routers_file(self, workflow_ir: WorkflowIR, agent_ir: AgentIR) -> str:
        """生成路由函数文件"""
        lines = [
            '"""',
            '路由函数',
            '"""',
            '',
            'from openjiuwen.core.runtime.workflow import WorkflowRuntime',
        ]

        # 收集路由函数中使用的外部变量
        router_code = '\n'.join(e.condition_func or '' for e in workflow_ir.edges if e.is_conditional)
        config_vars = self._find_config_vars_in_code(router_code, agent_ir)

        if config_vars:
            lines.append(f'from .config import {", ".join(config_vars)}')

        lines.append('')

        # 添加路由函数
        for edge in workflow_ir.edges:
            if edge.is_conditional and edge.condition_func:
                lines.append(edge.condition_func)
                lines.append('')

        if not any(e.is_conditional for e in workflow_ir.edges):
            lines.append('# 无条件路由')

        return '\n'.join(lines)

    def _find_config_vars_in_code(self, code: str, agent_ir: AgentIR) -> List[str]:
        """查找代码中使用的 config 变量"""
        import re
        config_vars = []

        # 从全局变量中提取变量名
        all_var_names = set()
        for var in agent_ir.global_vars:
            if '=' in var:
                var_name = var.split('=')[0].strip()
                all_var_names.add(var_name)

        # 检查代码中是否使用了这些变量
        for var_name in all_var_names:
            # 确保是独立的变量引用（不是字符串的一部分）
            if re.search(rf'\b{var_name}\b', code):
                config_vars.append(var_name)

        return config_vars

    def _gen_workflow_file(self, workflow_ir: WorkflowIR, agent_ir: AgentIR) -> str:
        """生成工作流构建文件"""
        lines = [
            '"""',
            '工作流构建',
            '"""',
            '',
            'from openjiuwen.core.workflow.base import Workflow',
            'from openjiuwen.core.component.start_comp import Start',
            'from openjiuwen.core.component.end_comp import End',
            '',
        ]

        # 导入组件
        for node in workflow_ir.nodes:
            lines.append(f'from .components.{node.name}_comp import {node.class_name}')

        # 导入路由
        router_names = [e.router_name for e in workflow_ir.edges if e.is_conditional and e.router_name]
        if router_names:
            lines.append(f'from .routers import {", ".join(router_names)}')

        lines.append('')
        lines.append('')

        # 生成工作流构建函数
        lines.append(self._gen_workflow_builder(workflow_ir, agent_ir))

        return '\n'.join(lines)

    def _gen_main_file(self, agent_ir: AgentIR, workflow_ir: Optional[WorkflowIR] = None) -> str:
        """生成主入口文件"""
        agent_name = agent_ir.name.lower()

        # 使用从源代码提取的示例输入
        input_fields = []

        if agent_ir.initial_inputs:
            for field_name, field_value in agent_ir.initial_inputs.items():
                # 优先使用 example_inputs 中的示例值
                if field_name in agent_ir.example_inputs:
                    example_value = agent_ir.example_inputs[field_name]
                    input_fields.append(f'        "{field_name}": {repr(example_value)}')
                elif isinstance(field_value, str) and field_value.startswith("${"):
                    # 变量占位符，没有示例值时使用空值
                    input_fields.append(f'        "{field_name}": ""')
                else:
                    # 直接值
                    input_fields.append(f'        "{field_name}": {repr(field_value)}')
        else:
            # 回退：空输入
            input_fields = ['        # TODO: 添加输入参数']

        inputs_content = ",\n".join(input_fields)

        return f'''"""
主入口
"""

import asyncio
import sys
import os

# 支持直接运行
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openjiuwen.core.runtime.workflow import WorkflowRuntime

from {agent_name}.workflow import build_{agent_name}_workflow


async def main():
    """主函数"""
    workflow = build_{agent_name}_workflow()
    runtime = WorkflowRuntime()

    # 示例输入
    inputs = {{
{inputs_content}
    }}

    result = await workflow.invoke(inputs, runtime)
    print("执行结果:", result)


def run(inputs: dict) -> dict:
    """运行 Agent"""
    workflow = build_{agent_name}_workflow()
    runtime = WorkflowRuntime()
    return asyncio.run(workflow.invoke(inputs, runtime))


if __name__ == "__main__":
    asyncio.run(main())
'''

    def _serialize_ir(
        self,
        agent_ir: AgentIR,
        workflow_ir: WorkflowIR,
        migration_ir: Optional[MigrationIR]
    ) -> Dict[str, Any]:
        """序列化 IR 为字典（完整版本）"""
        return {
            "agent": {
                "name": agent_ir.name,
                "llm_config": {
                    "model_name": agent_ir.llm_config.model_name,
                    "temperature": agent_ir.llm_config.temperature,
                    "other_params": agent_ir.llm_config.other_params
                } if agent_ir.llm_config else None,
                "tools": [
                    {
                        "name": t.name,
                        "func_name": t.func_name,
                        "description": t.description,
                        "parameters": t.parameters,
                        "converted_body": t.converted_body
                    }
                    for t in agent_ir.tools
                ],
                "state_fields": agent_ir.state_fields,
                "global_vars": agent_ir.global_vars,
                "tool_related_vars": agent_ir.tool_related_vars,
                "tool_map_var_name": agent_ir.tool_map_var_name,
                "initial_inputs": agent_ir.initial_inputs,
                "example_inputs": agent_ir.example_inputs
            },
            "workflow": {
                "entry_node": workflow_ir.entry_node,
                "state_class_name": workflow_ir.state_class_name,
                "nodes": [
                    {
                        "name": n.name,
                        "class_name": n.class_name,
                        "inputs": n.inputs,
                        "outputs": n.outputs,
                        "conversion_source": n.conversion_source,
                        "has_llm": n.has_llm,
                        "has_tools": n.has_tools,
                        "docstring": n.docstring,
                        "converted_body": n.converted_body
                    }
                    for n in workflow_ir.nodes
                ],
                "edges": [
                    {
                        "source": e.source,
                        "target": e.target,
                        "is_conditional": e.is_conditional,
                        "condition_func": e.condition_func,
                        "condition_map": e.condition_map,
                        "router_name": e.router_name
                    }
                    for e in workflow_ir.edges
                ]
            },
            "stats": migration_ir.conversion_stats if migration_ir else {}
        }

    def _gen_report(
        self,
        agent_ir: AgentIR,
        workflow_ir: WorkflowIR,
        migration_ir: Optional[MigrationIR],
        generated_files: List[str]
    ) -> str:
        """生成迁移报告"""
        stats = migration_ir.conversion_stats if migration_ir else {}

        report_lines = [
            "# LG2Jiuwen 迁移报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 概览",
            "",
            f"- **Agent 名称**: {agent_ir.name}",
            f"- **节点数量**: {len(workflow_ir.nodes)}",
            f"- **边数量**: {len(workflow_ir.edges)}",
            f"- **工具数量**: {len(agent_ir.tools)}",
            "",
            "## 转换统计",
            "",
            "| 指标 | 数量 |",
            "|------|------|",
            f"| 规则处理 | {stats.get('rule_count', 0)} |",
            f"| AI 处理 | {stats.get('ai_count', 0)} |",
            f"| 总节点数 | {stats.get('total_nodes', len(workflow_ir.nodes))} |",
            f"| 总边数 | {stats.get('total_edges', len(workflow_ir.edges))} |",
            f"| 总工具数 | {stats.get('total_tools', len(agent_ir.tools))} |",
            "",
            "## 节点详情",
            "",
            "| 节点名 | 类名 | 输入 | 输出 | 转换来源 |",
            "|--------|------|------|------|----------|",
        ]

        for node in workflow_ir.nodes:
            inputs_str = ", ".join(node.inputs) if node.inputs else "-"
            outputs_str = ", ".join(node.outputs) if node.outputs else "-"
            report_lines.append(
                f"| {node.name} | {node.class_name} | {inputs_str} | {outputs_str} | {node.conversion_source} |"
            )

        report_lines.extend([
            "",
            "## 边详情",
            "",
            "| 源节点 | 目标节点 | 类型 |",
            "|--------|----------|------|",
        ])

        for edge in workflow_ir.edges:
            edge_type = "条件" if edge.is_conditional else "普通"
            target = edge.target or f"(路由: {edge.router_name})"
            report_lines.append(f"| {edge.source} | {target} | {edge_type} |")

        if agent_ir.tools:
            report_lines.extend([
                "",
                "## 工具详情",
                "",
                "| 工具名 | 描述 |",
                "|--------|------|",
            ])
            for tool in agent_ir.tools:
                report_lines.append(f"| {tool.name} | {tool.description} |")

        report_lines.extend([
            "",
            "## 生成的文件",
            "",
        ])
        for f in generated_files:
            report_lines.append(f"- `{f}`")

        report_lines.extend([
            "",
            "---",
            "*由 LG2Jiuwen 自动生成*"
        ])

        return "\n".join(report_lines)

    def _gen_imports(self, agent_ir: AgentIR, workflow_ir: WorkflowIR) -> str:
        """生成导入语句"""
        imports = [
            '"""',
            f'{agent_ir.name} - 由 LG2Jiuwen 自动迁移生成',
            '"""',
            '',
            'import os',
            'import asyncio',
            'from typing import Any, Dict, List, Optional',
            '',
            'from openjiuwen.core.workflow.base import Workflow',
            'from openjiuwen.core.component.start_comp import Start',
            'from openjiuwen.core.component.end_comp import End',
            'from openjiuwen.core.component.base import WorkflowComponent',
            'from openjiuwen.core.runtime.base import ComponentExecutable, Input, Output',
            'from openjiuwen.core.runtime.runtime import Runtime',
            'from openjiuwen.core.runtime.workflow import WorkflowRuntime',
            'from openjiuwen.core.context_engine.base import Context',
        ]

        # 如果有 LLM 调用
        has_llm = any(node.has_llm for node in workflow_ir.nodes)
        if has_llm:
            imports.append('from openjiuwen.core.utils.llm.model_library.openai import OpenAIChatModel')

        # 如果有工具
        if agent_ir.tools:
            imports.append('from openjiuwen.core.utils.tool.param import Param')
            imports.append('from openjiuwen.core.utils.tool.tool import tool')

        # 添加原始代码中的必要导入
        tool_code = "\n".join(t.converted_body for t in agent_ir.tools)
        node_code = "\n".join(n.converted_body for n in workflow_ir.nodes)
        all_code = tool_code + node_code

        # 检查并添加常用依赖
        if "httpx" in all_code:
            imports.insert(4, 'import httpx')

        result = '\n'.join(imports)

        # 默认添加环境变量
        result += "\nos.environ['LLM_SSL_VERIFY'] = 'false'"

        # 添加全局变量
        if agent_ir.global_vars:
            result += '\n\n'
            result += '\n'.join(agent_ir.global_vars)

        return result

    def _gen_tool(self, tool: ToolIR) -> str:
        """生成工具函数"""
        # 生成 Param 列表
        param_defs = []
        func_params = []
        for p in tool.parameters:
            param_name = p["name"]
            param_type = p.get("type", "Any")
            # 转换类型为 Param 的 type 字符串
            type_map = {
                "str": "string",
                "int": "integer",
                "float": "number",
                "bool": "boolean",
                "list": "array",
                "dict": "object",
            }
            param_type_str = type_map.get(param_type, "string")
            param_defs.append(
                f'        Param(name="{param_name}", description="{param_name}", type="{param_type_str}", required=True)'
            )
            # 函数签名使用原始 Python 类型
            func_params.append(f'{param_name}: {param_type}')

        params_list = ",\n".join(param_defs) if param_defs else ""
        func_params_str = ", ".join(func_params) if func_params else ""

        # 使用新格式生成工具
        if params_list:
            return f'''@tool(
    name="{tool.func_name}",
    description="{tool.description}",
    params=[
{params_list}
    ]
)
def {tool.func_name}({func_params_str}) -> str:
    """{tool.description}"""
{self._indent(tool.converted_body, 4)}'''
        else:
            return f'''@tool(
    name="{tool.func_name}",
    description="{tool.description}"
)
def {tool.func_name}() -> str:
    """{tool.description}"""
{self._indent(tool.converted_body, 4)}'''

    def _gen_tool_map_and_invoke(self, agent_ir: AgentIR) -> str:
        """生成工具映射和 invoke_tool 函数（单文件模式）"""
        lines = []
        tool_map_name = agent_ir.tool_map_var_name or "tool_map"

        # 生成工具映射
        if agent_ir.tool_related_vars:
            lines.append('# 工具映射')
            for var in agent_ir.tool_related_vars:
                lines.append(var)
        elif agent_ir.tools:
            lines.append('# 工具映射')
            tool_entries = ', '.join(
                f'"{t.name}": {t.func_name}' for t in agent_ir.tools
            )
            lines.append(f'{tool_map_name} = {{{tool_entries}}}')

        # 生成 invoke_tool 函数
        lines.extend([
            '',
            '',
            'def invoke_tool(tool_name: str, arg: str) -> str:',
            '    """',
            '    调用工具的辅助函数',
            '',
            '    openJiuwen 的 @tool 装饰器返回 LocalFunction，',
            '    需要通过 .invoke(inputs={param_name: arg}) 调用。',
            '    此函数自动处理参数名映射。',
            '    """',
            f'    tool_func = {tool_map_name}.get(tool_name)',
            '    if tool_func is None:',
            '        return f"未知工具: {tool_name}"',
            '    # 获取工具的第一个参数名',
            '    if hasattr(tool_func, "params") and tool_func.params:',
            '        param_name = tool_func.params[0].name',
            '    else:',
            '        param_name = "input"',
            '    return tool_func.invoke(inputs={param_name: arg})',
        ])

        return '\n'.join(lines)

    def _gen_component(self, node: WorkflowNodeIR, agent_ir: AgentIR) -> str:
        """生成组件类

        数据传递方式：
        1. return 的值自动同步给下游，存储为 {节点名}.{字段名}
        2. 全局状态变量需显式调用 runtime.update_global_state() 更新，
           所有组件都能通过 runtime.get_global_state("key") 访问
        """
        # 生成初始化方法
        init_method = self._gen_init_method(node, agent_ir)

        # 生成输出初始化
        output_init = self._gen_output_init(node.outputs)

        # 生成输出字典（所有输出通过 return 传递给下游）
        outputs_dict = "{" + ", ".join(f'"{o}": {o}' for o in node.outputs) + "}" if node.outputs else "{}"

        # 确定哪些是全局状态变量（初始输入）
        global_state_keys = set(agent_ir.initial_inputs.keys()) if agent_ir.initial_inputs else set()
        global_outputs = [o for o in node.outputs if o in global_state_keys]

        # 处理组件逻辑代码
        body_code = node.converted_body
        body_code = body_code.replace("__COLLECTED_OUTPUTS__", outputs_dict)

        # 将全局状态变量的访问从 inputs.get/inputs[] 转换为 runtime.get_global_state()
        body_code = self._convert_global_state_access(body_code, global_state_keys)

        docstring = node.docstring or f"{node.name} 组件"

        # 检查代码是否已经以 return 结尾
        body_lines = [l for l in body_code.strip().split('\n') if l.strip()]
        has_final_return = body_lines and body_lines[-1].strip().startswith('return ')

        # 生成更新全局状态的代码
        if global_outputs:
            global_dict = "{" + ", ".join(f'"{o}": {o}' for o in global_outputs) + "}"
            update_global_state = f"runtime.update_global_state({global_dict})"
        else:
            update_global_state = ""

        if has_final_return:
            # 在 return 之前插入 update_global_state
            if update_global_state:
                body_code = self._insert_global_state_update(body_code, global_outputs)
            return f'''class {node.class_name}(WorkflowComponent, ComponentExecutable):
    """{docstring}"""

{init_method}

    async def invoke(
        self,
        inputs: Input,
        runtime: Runtime,
        context: Context
    ) -> Output:
{output_init}
        # 组件逻辑（转换来源: {node.conversion_source}）
{self._indent(body_code, 8)}'''
        else:
            ending_code = ""
            if update_global_state:
                ending_code = f"\n        # 更新全局状态\n        {update_global_state}"

            return f'''class {node.class_name}(WorkflowComponent, ComponentExecutable):
    """{docstring}"""

{init_method}

    async def invoke(
        self,
        inputs: Input,
        runtime: Runtime,
        context: Context
    ) -> Output:
{output_init}
        # 组件逻辑（转换来源: {node.conversion_source}）
{self._indent(body_code, 8)}{ending_code}
        return {outputs_dict}'''

    def _gen_init_method(self, node: WorkflowNodeIR, agent_ir: AgentIR) -> str:
        """生成初始化方法"""
        if node.has_llm:
            model_name = agent_ir.llm_config.model_name if agent_ir.llm_config else "gpt-4"
            # 从 agent_ir 获取 API 配置
            api_key = "os.getenv('OPENAI_API_KEY', '')"
            api_base = "os.getenv('OPENAI_API_BASE', '')"
            if agent_ir.llm_config and agent_ir.llm_config.other_params:
                if "api_key" in agent_ir.llm_config.other_params:
                    api_key = f'"{agent_ir.llm_config.other_params["api_key"]}"'
                if "api_base" in agent_ir.llm_config.other_params:
                    api_base = f'"{agent_ir.llm_config.other_params["api_base"]}"'
            return f'''    def __init__(self, llm=None):
        if llm:
            self._llm = llm
        else:
            self._llm = OpenAIChatModel(
                api_key={api_key},
                api_base={api_base}
            )
        self.model_name = "{model_name}"'''
        else:
            return '''    def __init__(self):
        pass'''

    def _gen_output_init(self, outputs: List[str]) -> str:
        """生成输出变量初始化"""
        if not outputs:
            return "        # 无输出变量"

        lines = ["        # 初始化输出变量"]
        for output in outputs:
            lines.append(f"        {output} = None")
        return "\n".join(lines)

    def _gen_workflow_builder(self, workflow_ir: WorkflowIR, agent_ir: AgentIR) -> str:
        """生成工作流构建函数"""
        lines = [
            f'def build_{agent_ir.name.lower()}_workflow() -> Workflow:',
            f'    """构建 {agent_ir.name} 工作流"""',
            '',
            '    workflow = Workflow()',
            '',
            '    # 设置起点',
            '    workflow.set_start_comp("start", Start(), inputs_schema={',
        ]

        # 添加输入 schema - 使用从源代码 invoke() 提取的初始输入
        if agent_ir.initial_inputs:
            for field_name in agent_ir.initial_inputs.keys():
                lines.append(f'        "{field_name}": "${{{field_name}}}",')
        else:
            # 回退：使用入口节点的输入字段
            if workflow_ir.entry_node:
                entry_node = workflow_ir.get_node_by_name(workflow_ir.entry_node)
                if entry_node:
                    for field_name in entry_node.inputs:
                        lines.append(f'        "{field_name}": "${{{field_name}}}",')
        lines.append('    })')
        lines.append('')

        # 添加组件
        lines.append('    # 添加组件')
        for node in workflow_ir.nodes:
            # 构建输入 schema
            inputs_schema = self._build_inputs_schema(node, workflow_ir)
            lines.append(f'    workflow.add_workflow_comp(')
            lines.append(f'        "{node.name}",')
            lines.append(f'        {node.class_name}(),')
            lines.append(f'        inputs_schema={{{inputs_schema}}}')
            lines.append('    )')
            lines.append('')

        # 设置终点 - 收集所有节点的输出
        end_inputs_schema = self._build_end_inputs_schema(workflow_ir, agent_ir)
        lines.append('    # 设置终点')
        lines.append(f'    workflow.set_end_comp("end", End(), inputs_schema={{{end_inputs_schema}}})')
        lines.append('')

        # 添加连接
        lines.append('    # 添加连接')
        if workflow_ir.entry_node:
            lines.append(f'    workflow.add_connection("start", "{workflow_ir.entry_node}")')

        for edge in workflow_ir.edges:
            if edge.is_conditional:
                # 条件连接 - openJiuwen 只需要源组件ID和路由函数两个参数
                lines.append(f'    workflow.add_conditional_connection("{edge.source}", {edge.router_name})')
            else:
                lines.append(f'    workflow.add_connection("{edge.source}", "{edge.target}")')

        lines.append('')
        lines.append('    return workflow')

        return '\n'.join(lines)

    def _build_inputs_schema(self, node: WorkflowNodeIR, workflow_ir: WorkflowIR) -> str:
        """构建节点的输入 schema"""
        schema_parts = []
        for input_field in node.inputs:
            # 尝试从前置节点找
            source_node = self._find_source_for_field(input_field, node.name, workflow_ir)
            if source_node:
                schema_parts.append(f'"{input_field}": "${{{source_node}.{input_field}}}"')
            else:
                schema_parts.append(f'"{input_field}": "${{start.{input_field}}}"')

        return ", ".join(schema_parts)

    def _build_end_inputs_schema(self, workflow_ir: WorkflowIR, agent_ir: AgentIR) -> str:
        """构建 End 组件的输入 schema，收集所有重要输出"""
        schema_parts = []
        collected_fields = set()

        # 从所有节点的输出中收集字段
        for node in workflow_ir.nodes:
            for output_field in node.outputs:
                if output_field not in collected_fields:
                    collected_fields.add(output_field)
                    schema_parts.append(f'"{output_field}": "${{{node.name}.{output_field}}}"')

        return ", ".join(schema_parts)

    def _find_source_for_field(
        self,
        field: str,
        target_node: str,
        workflow_ir: WorkflowIR
    ) -> Optional[str]:
        """找到字段的来源节点"""
        # 1. 查找直接指向 target_node 的普通边
        for edge in workflow_ir.edges:
            if edge.target == target_node and not edge.is_conditional:
                source_node = workflow_ir.get_node_by_name(edge.source)
                if source_node and field in source_node.outputs:
                    return edge.source

        # 2. 查找条件边，检查其 condition_map 是否包含 target_node
        for edge in workflow_ir.edges:
            if edge.is_conditional and edge.condition_map:
                if target_node in edge.condition_map.values():
                    source_node = workflow_ir.get_node_by_name(edge.source)
                    if source_node and field in source_node.outputs:
                        return edge.source

        # 3. 遍历所有节点，查找输出该字段的节点（作为后备）
        for n in workflow_ir.nodes:
            if n.name != target_node and field in n.outputs:
                return n.name

        return None

    def _gen_main(self, agent_ir: AgentIR) -> str:
        """生成主函数（单文件模式）"""
        # 使用从源代码提取的示例输入
        input_fields = []

        if agent_ir.initial_inputs:
            for field_name, field_value in agent_ir.initial_inputs.items():
                # 优先使用 example_inputs 中的示例值
                if field_name in agent_ir.example_inputs:
                    example_value = agent_ir.example_inputs[field_name]
                    input_fields.append(f'        "{field_name}": {repr(example_value)}')
                elif isinstance(field_value, str) and field_value.startswith("${"):
                    # 变量占位符，没有示例值时使用空值
                    input_fields.append(f'        "{field_name}": ""')
                else:
                    # 直接值
                    input_fields.append(f'        "{field_name}": {repr(field_value)}')
        else:
            # 回退：空输入
            input_fields = ['        # TODO: 添加输入参数']

        inputs_content = ",\n".join(input_fields)

        return f'''async def main():
    """主函数"""
    workflow = build_{agent_ir.name.lower()}_workflow()
    runtime = WorkflowRuntime()

    # 示例输入
    inputs = {{
{inputs_content}
    }}

    result = await workflow.invoke(inputs, runtime)
    print("执行结果:", result)


if __name__ == "__main__":
    asyncio.run(main())'''

    def _indent(self, code: str, spaces: int) -> str:
        """缩进代码，保持相对缩进"""
        if not code:
            return " " * spaces + "pass"

        lines = code.split("\n")

        # 找出非空行的最小缩进
        min_indent = float('inf')
        for line in lines:
            if line.strip():  # 非空行
                leading = len(line) - len(line.lstrip())
                min_indent = min(min_indent, leading)

        if min_indent == float('inf'):
            min_indent = 0

        # 重新缩进
        indented = []
        for line in lines:
            if line.strip():
                # 移除原有的最小缩进，添加目标缩进
                stripped = line[min_indent:] if len(line) >= min_indent else line.lstrip()
                indented.append(" " * spaces + stripped)
            else:
                indented.append("")
        return "\n".join(indented)

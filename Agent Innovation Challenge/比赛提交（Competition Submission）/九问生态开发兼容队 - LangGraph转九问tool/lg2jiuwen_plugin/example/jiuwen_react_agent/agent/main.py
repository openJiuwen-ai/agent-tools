"""
主入口
"""

import asyncio
import sys
import os

# 支持直接运行
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openjiuwen.core.runtime.workflow import WorkflowRuntime

from agent.workflow import build_agent_workflow


async def main():
    """主函数"""
    workflow = build_agent_workflow()
    runtime = WorkflowRuntime()

    # 示例输入
    inputs = {
        "input": '今天天气？',
        "is_end": False,
        "loop_count": 0
    }

    result = await workflow.invoke(inputs, runtime)
    print("执行结果:", result)


def run(inputs: dict) -> dict:
    """运行 Agent"""
    workflow = build_agent_workflow()
    runtime = WorkflowRuntime()
    return asyncio.run(workflow.invoke(inputs, runtime))


if __name__ == "__main__":
    asyncio.run(main())

"""
主入口
"""

import sys
import os

# 支持直接运行
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from react_agent.graph import app


def run(input_text: str) -> dict:
    """运行 Agent"""
    return app.invoke({
        "input": input_text,
        "is_end": False,
        "loop_count": 0
    })


if __name__ == "__main__":

    input_text = "100加200等于多少？"
    print(f"\n{'='*50}")
    print(f"输入: {input_text}")
    result = run(input_text)
    print(f"选择工具: {result.get('selected_tool')}")
    print(f"工具参数: {result.get('tool_input')}")
    print(f"思考: {result['thought']}")
    print(f"结果: {result['result']}")
    print(f"循环次数: {result.get('loop_count', 0)}")

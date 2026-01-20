import json
from openjiuwen.core.utils.llm.model_utils.model_factory import ModelFactory
import re
import os
from autoCursor_Tools import *
from dotenv import load_dotenv
import math
from PIL import Image

# tools_schema = [screenshot.get_tool_info(), dragTo.get_tool_info(), getPosition.get_tool_info(), moveTo.get_tool_info(),
#                 write.get_tool_info(), click.get_tool_info(), doubleClick.get_tool_info(), scroll.get_tool_info(), findInteractionPixel.get_tool_info()]

os.environ["LLM_SSL_VERIFY"] = "false"

tools_schema = [
    screenshot.get_tool_info(),
    dragTo.get_tool_info(),
    getPosition.get_tool_info(),
    moveTo.get_tool_info(),
    write.get_tool_info(),
    click.get_tool_info(),
    doubleClick.get_tool_info(),
    scroll.get_tool_info(),
    pressEnter.get_tool_info(),
    findInteractionPixel.get_tool_info(),
]

tool_map = {
    "screenshot": screenshot.func,
    "dragTo": dragTo.func,
    "getPosition": getPosition.func,
    "moveTo": moveTo.func,
    "write": write.func,
    "click": click.func,
    "doubleClick": doubleClick.func,
    "scroll": scroll.func,
    "pressEnter": pressEnter.func,
    "findInteractionPixel": findInteractionPixel.func
}

# 禁用 LLM SSL 验证
os.environ["LLM_SSL_VERIFY"] = "false"


# -----------------------------
# 参数解析 & 规范化
# -----------------------------
def normalize_int(v):
    """
    将模型输出的各种坐标、长度等值规范化为 int。
    支持：
    - list → 取第一个
    - "41, 559" 或 "41 559" → 取第一个
    - float / str float → int
    """
    if v is None:
        return None
    if isinstance(v, list):
        return normalize_int(v[0])
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", v.strip())
        if not nums:
            raise ValueError(f"无法解析为 int: {v}")
        return int(float(nums[0]))
    raise ValueError(f"不支持的参数类型: {type(v)}")


def parse_params(params: dict):
    # 特殊处理：如果 'x' 是 [x, y] 或 [x, y, w, h]
    if "x" in params and isinstance(params["x"], list):
        coord_list = params["x"]
        if len(coord_list) == 2:
            # 视为 (x, y)
            params["x"] = normalize_int(coord_list[0])
            params["y"] = normalize_int(coord_list[1])
        elif len(coord_list) >= 4:
            # 视为 (x, y, w, h)
            params["x"] = normalize_int(coord_list[0])
            params["y"] = normalize_int(coord_list[1])
            params["w"] = normalize_int(coord_list[2])
            params["h"] = normalize_int(coord_list[3])
        else:
            raise ValueError(f"坐标列表长度不支持: {coord_list}")
        # 注意：这里保留了 "x"，但已转为标量；其他字段如 y/w/h 会被后续覆盖或新增

    # 普通字段转 int
    for k in ["x", "y", "w", "h", "len"]:
        if k in params:
            try:
                params[k] = normalize_int(params[k])
            except (ValueError, TypeError) as e:
                raise ValueError(f"参数 '{k}' 格式错误: {e}")

    return params


# -----------------------------
# 系统 Prompt 构建
# -----------------------------
def build_system_prompt(has_screenshot: bool) -> str:
    return rf"""
        你是一个顶级的AI视觉操作代理。你的任务是分析电脑屏幕截图，理解用户的指令，然后将任务分解为单一、精确的GUI原子操作。只能通过调用工具执行操作，禁止直接输出文本结论。\
        作为GUI Agent，你处理界面的逻辑核心是 OODA 循环（观察 Observe、定位 Orient、决策 Decide、执行 Act）
        重要！必须先将任务进行拆解！！！并输出步骤\
        执行完一个步骤之后，默认这个步骤已经完成，继续一个步骤，不要在某一步一直卡着！！！！！！
        【当前状态】
        - has_screenshot = {has_screenshot}
        【当前截图状态】：
        - 最新截图有效 = {has_screenshot}
        - 如果最新截图无效（False），必须调用 screenshot(full=true)

        【核心原则】
        1. 所有视觉判断必须来自 screenshot
        2. 没有截图或不确定界面状态必须调用 screenshot(full=true)
        3. screenshot 永远合法


        【thought 规则】
        - 必须生成 thought（只一句话）
        - 内容示例： 「我在截图中看到 XXX，因此需要执行 YYY」
        - 第一次截图示例： 「我还没有看到屏幕内容，需要先截图以进行后续判断」

        【工具调用是唯一输出方式】
        - 每轮必须调用且只能调用一个工具
        - 禁止只输出文本

        【可用工具】
        ========================
        1. screenshot(x:int, y:int, w:int, h:int, full:bool)
           - 用途：获取屏幕图像
           - 规则：
             * 所有视觉判断必须来自 screenshot
             * 不确定时必须截图

        2. getPosition()
           - 获取当前鼠标位置

        3. moveTo(x:int, y:int)
            - 用途：移动鼠标到指定位置

        4. click(x:int, y:int, button:str='left')
            - 用途：单击鼠标，选中使用单击

        5. doubleClick(x:int, y:int, button:str='left')
            - 用途：双击鼠标，打开图标必须使用双击

        6. dragTo(x:int, y:int)
            - 用途：使用鼠标拖拽

        7. write(text:str)
            - 用途：键入文本

        8. scroll(len:int)
            - 用途：滚轮操作\
        9. pressEnter()\
            - 用途：按下回车键\
        10.findInteractionPixel(x:int, y:int)\
            - 用途：微调鼠标

        【执行流程】
        1. 没有 screenshot → screenshot(full=true)
        2. 基于最新 screenshot 做判断
        3. 如果执行了 click / doubleClick / write / scroll：
           - 下一步必须调用 screenshot 验证界面是否变化
           - 不能假设操作成功
        4. 只有看到目标元素（如浏览器地址栏），才能进行后续操作

        【绝对禁止】
        - 猜测坐标
        - 没有 screenshot 做视觉判断
        - 输出多个 action
        - 只输出 thought 不调用工具

        【最终硬规则】
        - 永远不能返回空
        - 永远不能没有 tool_call 结束
        - 无法决策 → 默认 screenshot(full=true)
        - has_screenshot=False → 必须 screenshot
        - has_screenshot=True → 禁止重复 screenshot
        - 所有坐标判断必须来自已有 screenshot
        - 每轮必须 tool_call\
        - 严禁假设操作成功！必须通过新截图确认界面状态变化后，才能进行下一步。

        """


def run_agent(user_query: str, max_steps: int = 3):
    # 1. 初始化模型
    factory = ModelFactory()
    load_dotenv()
    model = factory.get_model(
        model_provider="OpenAI",
        api_base=os.getenv("DATABASE_URL"),  # API url
        api_key=os.getenv("API_KEY")  # API Key
    )

    has_screenshot = False
    vl_high_resolution_images = False  # 与 invoke 参数一致

    messages = [
        {"role": "system", "content": build_system_prompt(has_screenshot)},
        {"role": "user", "content": user_query}
    ]

    for step in range(max_steps):
        messages[0]["content"] = build_system_prompt(has_screenshot)

        print(f"\n--- Step {step + 1} ---\n")

        # 调用模型
        response = model.invoke(
            model_name="qwen3-vl-plus",
            messages=messages,
            tools=tools_schema,
            temperature=0.1,
            top_p=0.95,
            extra_body={"vl_high_resolution_images": vl_high_resolution_images}
        )

        print("LLM Response:", response)

        assistant_msg = {
            "role": "assistant",
            "content": response.content or "",
        }
        if response.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments}
                } for tc in response.tool_calls
            ]
        messages.append(assistant_msg)

        # 解析工具调用
        tool_calls_to_execute = []
        if response.tool_calls:
            for tc in response.tool_calls:
                try:
                    args = json.loads(tc.arguments) if tc.arguments else {}
                except:
                    args = {}
                args = parse_params(args)
                tool_calls_to_execute.append((tc.name, args))
        else:
            try:
                content_str = response.content.strip()
                if content_str.startswith("{") and content_str.endswith("}"):
                    action_dict = json.loads(content_str)
                    action_name = action_dict.get("action")
                    params = parse_params(action_dict.get("parameters", {}))
                    tool_calls_to_execute.append((action_name, params))
                else:
                    print("未解析到可执行 action")
                    return response.content
            except Exception as e:
                print("解析 content JSON 失败:", e)

        # 执行工具调用
        for action_name, params in tool_calls_to_execute:
            if (
                    action_name in ["click", "doubleClick", "moveTo", "dragTo"]
                    and "x" in params
                    and "y" in params
            ):
                try:

                    params["x"] = params["x"] / 1000 * 1920
                    params["y"] = params["y"] / 1000 * 1080
                    print("转换后：", {"x": params["x"], "y": params["y"]})
                except Exception as e:
                    print("smart_size 坐标换算失败:", e)
                    continue
            if action_name in tool_map:
                result = tool_map[action_name](**params)
                if action_name == "screenshot":
                    has_screenshot = True
                    tool_content = [
                        {"type": "text", "text": "这是最新的全屏截图,分析这截图，然后用于决定下一步行动。"},
                        {"type": "image_url", "image_url": {"url": result}
                         }
                    ]
                    print("screenshot captured")
                else:
                    has_screenshot = False
                    tool_content = json.dumps(
                        f"工具 {action_name} 执行完成，结果:{result},进行下一步操作前请截图，以便决定下一步的操作",
                        ensure_ascii=False)
                    time.sleep(1.0)  # 等待界面响应
                    print(f"工具 {action_name} 执行完成，结果:", tool_content)
            else:
                tool_content = f"未知工具: {action_name}"
                print(f" {tool_content}")

            # 回传工具结果给模型
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,  # 必须匹配！
                "name": action_name,
                "content": tool_content
            })

        # 回传 assistant 消息
        assistant_message = {
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments
                    }
                }
                for tc in response.tool_calls
            ] if response.tool_calls else None
        }
        messages.append(assistant_message)
    return "Done"


# -----------------------------
# 主程序入口
# -----------------------------
if __name__ == "__main__":
    print("等待5秒，请切换到目标窗口...")

    time.sleep(5)
    x = 718
    y = 439
    print((x,y))
    query = f"在({x, y})附近微调鼠标"

    run_agent(query, max_steps=5)

import os
import json
import time
import re
from dotenv import load_dotenv
from openjiuwen.core.utils.llm.model_utils.model_factory import ModelFactory

# =============================
# GUI 工具
# =============================
from autoCursor_Tools import (
    screenshot, click, doubleClick, moveTo,
    dragTo, scroll, write, getPosition, pressEnter, findInteractionPixel
)

# =============================
# 工具注册
# =============================
TOOLS_SCHEMA = [
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

TOOL_MAP = {
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

os.environ["LLM_SSL_VERIFY"] = "false"


# =============================
# 参数规范化
# =============================
def normalize_int(v):
    if v is None:
        return None
    if isinstance(v, list):
        return normalize_int(v[0])
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", v)
        if not nums:
            raise ValueError(f"无法解析参数: {v}")
        return int(float(nums[0]))
    raise ValueError(f"不支持的类型: {type(v)}")


def parse_params(params: dict):
    if "x" in params and isinstance(params["x"], list):
        arr = params["x"]
        params["x"] = normalize_int(arr[0])
        params["y"] = normalize_int(arr[1])
        if len(arr) >= 4:
            params["w"] = normalize_int(arr[2])
            params["h"] = normalize_int(arr[3])

    for k in ["x", "y", "w", "h", "len"]:
        if k in params:
            params[k] = normalize_int(params[k])

    return params


def smart_xy(x, y):
    """归一化坐标 → 屏幕坐标"""
    return int(x / 1000 * 1920), int(y / 1000 * 1080)


def normalize_xy(x, y):
    """ 屏幕坐标 -> 归一化坐标"""
    return int(x / 1920 * 1000), int(y / 1080 * 1000)


from typing import Dict, Any, Optional


def parse_xy_from_tool_args(arg_str: str) -> dict:
    """
    从 tool arguments 字符串中直接提取 x 和 y 的第一个整数
    支持各种破损的 JSON / 列表 / 字符串

    返回：
        {"x": int, "y": int}
    """
    arg_str = f'{arg_str}'
    if not arg_str or not isinstance(arg_str, str):
        raise ValueError("arguments 为空或不是字符串")

    # 使用正则找到 "x" 后的所有数字
    x_nums = re.findall(r'"?x"?\s*[:=]?\s*\[?\s*(-?\d+)', arg_str)
    y_nums = re.findall(r'"?y"?\s*[:=]?\s*\[?\s*(-?\d+)', arg_str)
    if not x_nums or not y_nums:
        # 万一没找到 x/y，就抓前两个数字
        nums = re.findall(r"-?\d+", arg_str)
        if len(nums) >= 2:
            return {"x": int(nums[0]), "y": int(nums[1])}
        raise ValueError(f"无法从 arguments 中解析 x/y: {arg_str}")

    return {"x": int(x_nums[0]), "y": int(y_nums[0])}


def build_system_prompt(has_screenshot: bool) -> str:
    return f"""
你是一个【桌面 GUI 视觉操作 Agent】。

你的职责只有一件事：
基于 screenshot 判断当前界面，并执行【一个】GUI 原子操作。

========================
【能力边界】
========================
- 你不能控制流程
- 你不能等待时间
- 你不能假设操作成功
- 你不能记住历史步骤

========================
【thought 规则（必须）】
========================
- 必须输出一句话 thought
- 示例：
  - 没截图：「我还没有看到屏幕，需要先截图判断界面」
  - 有截图：「我在截图中看到 XXX，因此需要执行 YYY」

========================
【工具规则】
========================
- 每轮必须且只能调用一个工具
- 所有判断必须来自 screenshot
- 无法判断 → screenshot(full=true)

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
    - 用途：滚轮操作
9. pressEnter()
    - 用途：按下回车键
10.findInteractionPixel(x:int, y:int)
    - 用途：微调鼠标

========================
【截图规则】
========================
- has_screenshot = False → 必须 screenshot
- 执行 click / write / scroll / findInteractionPixel 后 → 下一轮必须 screenshot
- 严禁假设界面已变化

========================
【禁止】
========================
- 猜测坐标
- 一次输出多个 action
- 只输出文本不调用工具

当前 has_screenshot = {has_screenshot}
"""


# =============================
# Agent 执行器（单阶段）
# =============================
def run_agent(user_query: str, max_steps: int = 3):
    load_dotenv()
    factory = ModelFactory()
    model = factory.get_model(
        model_provider="OpenAI",
        api_base=os.getenv("DATABASE_URL"),
        api_key=os.getenv("API_KEY")
    )

    has_screenshot = False

    messages = [
        {"role": "system", "content": build_system_prompt(has_screenshot)},
        {"role": "user", "content": user_query}
    ]

    for step in range(max_steps):
        messages[0]["content"] = build_system_prompt(has_screenshot)

        response = model.invoke(
            model_name="qwen3-vl-plus",
            messages=messages,
            tools=TOOLS_SCHEMA,
            temperature=0.1,
            top_p=0.95,
            extra_body={"vl_high_resolution_images": False}
        )
        print("LLM Response:", response)
        assistant_msg = {"role": "assistant", "content": response.content or ""}
        if response.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments}
                } for tc in response.tool_calls
            ]
        messages.append(assistant_msg)

        if not response.tool_calls:
            print("无工具调用，强制截图")
            response.tool_calls = [{
                "id": "fallback",
                "name": "screenshot",
                "arguments": json.dumps({"full": True})
            }]

        for tc in response.tool_calls:
            if tc == []:
                continue
            try:
                print(tc.name, tc.arguments)
            except Exception as e:
                print(e)
            if tc.name in ["click", "doubleClick", "moveTo", "dragTo", "findInteractionPixel"]:
                args = parse_xy_from_tool_args(tc.arguments)
            else:
                args = parse_params(json.loads(tc.arguments)) if tc.arguments else {}

            if tc.name in ["click", "doubleClick", "moveTo", "dragTo", "findInteractionPixel"] and "x" in args:
                args["x"], args["y"] = smart_xy(args["x"], args["y"])
                print("转化之后的坐标：", args["x"], args["y"])
            result = TOOL_MAP[tc.name](**args)

            if tc.name == "screenshot":
                has_screenshot = True
                tool_content = [
                    {"type": "text", "text": "最新截图，请基于该截图判断下一步"},
                    {"type": "image_url", "image_url": {"url": result}}
                ]
            else:
                has_screenshot = False
                tool_content = f"{tc.name} 执行完成"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": tool_content
            })

            time.sleep(0.8)

    return


import pyautogui
import numpy as np


def merge_nearby_boxes(boxes, distance_threshold=30):
    """
    合并中心点距离小于 threshold 的框
    :param boxes: list of Box objects from locateAllOnScreen
    :param distance_threshold: 中心点最大允许距离（像素）
    :return: 合并后的 boxes 列表
    """
    if not boxes:
        return []

    # 转换为 (center_x, center_y, box) 列表
    centers = []
    for box in boxes:
        cx, cy = pyautogui.center(box)
        centers.append((cx, cy, box))

    merged = []
    used = [False] * len(centers)

    for i, (cx1, cy1, box1) in enumerate(centers):
        if used[i]:
            continue
        cluster = [box1]
        used[i] = True
        for j in range(i + 1, len(centers)):
            if used[j]:
                continue
            cx2, cy2, box2 = centers[j]
            dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            if dist <= distance_threshold:
                cluster.append(box2)
                used[j] = True

        # 可选：取 cluster 中第一个，或计算平均位置
        # 这里我们取第一个（也可以用平均）
        merged.append(cluster[0])

    return merged


# =============================
# 主流程（CT 扫描 Demo）
# =============================
import random

if __name__ == "__main__":
    print("5 秒后开始，请切换到 CT 软件窗口")
    time.sleep(5)

    # ===== 模拟 CT 扫描步骤（Python 控制）=====
    ct_steps = [
        {"name": "localizer", "time": 5},
        {"name": "t2_lse_tra_448_pat2", "time": 84},
        {"name": "t1_se_tra", "time": 46},
        {"name": "t2_tirm_tra_dark-fl_pat2", "time": 84},
        {"name": "t2_lse_sag_320_pat2", "time": 120},
        {"name": "ep2d_diff_3scan_trace_p2", "time": 84}
    ]

    for idx, step in enumerate(ct_steps):

        print(f"\n▶ 正在执行步骤：{step['name']}，预计耗时：{step['time']} 秒")

        print(f"点击步骤 {step['name']}")
        run_agent(
            f"在界面中找到并点击扫描步骤 {step['name']}",
            max_steps=3
        )
        if idx == 1:
            exam_pos = pyautogui.locateAllOnScreen('right_top.jpg', confidence=0.8)
            exam_pos = list(exam_pos)
            exam_pos = merge_nearby_boxes(exam_pos)

            boxes_x = []
            for i, loc in enumerate(exam_pos):
                x, y, w, h = loc
                x = x + w
                boxes_x.append(x)
            boxes_x.sort()
            print(f"找到right_top共计{len(exam_pos)}个", boxes_x)
            width = boxes_x[1] - boxes_x[0]


        if idx != 0:
            locations = pyautogui.locateAllOnScreen('left.jpg', confidence=0.8)
            locations_list = list(locations)
            locations_list = merge_nearby_boxes(locations_list)
            print(f"找到垂直的点共计{len(exam_pos)}个")
            number = random.randint(1, len(locations_list))
            pixs = random.randint(10, 100)
            x, y = pyautogui.center(locations_list[number - 1])
            print(f"移动点{x, y}")
            move_left = False
            for i in range(len(boxes_x)):
                if boxes_x[i] - width <= x <= boxes_x[i] and boxes_x[i] - width <= x - pixs <= boxes_x[i]:
                    move_left = True
                    break
            if move_left:
                run_agent(
                    f"""
                        - 将鼠标移动到目标坐标{normalize_xy(x, y)}
                        - 鼠标在目标位置进行微调
                        - 从该点拖到{normalize_xy(x - pixs, y)}位置
                    """,
                    max_steps=4
                )

        print("点击 Sequence Ready 的 √ 按钮")
        run_agent(
            "查找并点击 Sequence Ready 的 √ 按钮",
            max_steps=3
        )
        print("点击底部的播放按钮")
        if idx == 0:
            run_agent(
                "查找并点击底部的播放按钮 ▶",
                max_steps=3
            )
        if idx != 0:
            print("确认弹窗")
            run_agent(
                "如果出现确认弹窗，点击 不调整 或 继续保存 按钮；若没有，仅截图",
                max_steps=3
            )

        print(f"等待 {step['time']} 秒...")
        time.sleep(step["time"])

    print(" CT 扫描流程执行完成")

import asyncio
import pyautogui
import time
import base64
import re
import os
from openjiuwen.core.utils.llm.model_utils.model_factory import ModelFactory
from io import BytesIO
from dotenv import load_dotenv

os.environ["LLM_SSL_VERIFY"] = "false"
os.environ["LLM_SSL_CERT"] = ""

# 扫雷游戏网页版链接：https://www.minesweeper.cn/#google_vignette
GAME_TOP_LEFT = (577, 264)  # 游戏窗口左上角坐标
GAME_BOARD_SIZE = (750, 400)  # 游戏棋盘区域的大小 (宽, 高)
CUBE_SIZE = 25
GAME_REGION = (*GAME_TOP_LEFT, *GAME_BOARD_SIZE)
print(GAME_REGION)

load_dotenv()
MODEL_NAME = "qwen-vl-plus"
API_BASE = os.getenv("DATABASE_URL") # API url
API_KEY = os.getenv("API_KEY")  # API Key


def screenshot_to_data_url(region):
    screenshot = pyautogui.screenshot(region=region)
    buffered = BytesIO()
    screenshot.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"


def parse_agent_response(response_obj):
    if hasattr(response_obj, 'content'):
        response_str = response_obj.content  # 从AIMessage对象取文本
    elif isinstance(response_obj, str):
        response_str = response_obj
    else:
        print(f"[警告] 不支持的返回类型: {type(response_obj)}")
        return None

    # 第二步：解析坐标
    if not response_str or response_str.strip() == '[]':
        return None
    match = re.search(r'\[(\d+),\s*(\d+)\]', response_str.strip())
    if match:
        row = int(match.group(1))
        col = int(match.group(2))
        return (row, col)
    print(f"[警告] 无法解析的响应内容: {response_str}")
    return None


def grid_to_screen_coords(grid_coords):
    row, col = grid_coords
    # 计算格子中心坐标（修正偏移逻辑）
    x = GAME_TOP_LEFT[0] + col * CUBE_SIZE - CUBE_SIZE // 2
    y = GAME_TOP_LEFT[1] + row * CUBE_SIZE - CUBE_SIZE // 2
    return (x, y)


# --- 5. AI Agent 核心函数 ---
async def get_safe_move_from_ai(model, image_data_url):
    print("--- 正在调用AI分析棋盘... ---")

    # 构建符合通义千问VL要求的messages结构
    messages = [
        {"role": "system", "content": """
           你是一个精通扫雷游戏的AI专家。你的任务是分析提供的扫雷棋盘图像，并找出一个绝对安全的格子进行点击。

           **你的分析步骤必须是：**
           1. 识别棋盘布局（行数和列数）；
           2. 识别每个格子的状态（数字0-8/已翻开/地雷）；
           3. 用扫雷规则推理100%安全的格子；
           4. 输出结果：仅返回 `[row, column]` 格式的坐标，无其他文字；若无安全格子返回 `[]`。
           """},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": "分析这张扫雷棋盘图片，找出一个安全的格子"}
            ]
        }
    ]

    try:
        response = await model.ainvoke(model_name=MODEL_NAME, messages=messages)
        print(f"AI 返回原始结果: {response}")
        return response
    except Exception as e:
        print(f"--- 调用AI失败 ---")
        print(f"错误详情: {e}")
        return "[]"


# --- 6. 主游戏循环 ---
async def main():
    print("扫雷AI Agent 已启动！")
    print("请确保扫雷游戏窗口已准备好，并位于指定位置。")
    time.sleep(5)

    # 初始化模型
    factory = ModelFactory()
    model = factory.get_model(
        model_provider="OpenAI",
        api_base=API_BASE,
        api_key=API_KEY
    )

    # 首次点击（棋盘中间）
    initial_x = GAME_TOP_LEFT[0] + GAME_BOARD_SIZE[0] // 2 - CUBE_SIZE // 2
    initial_y = GAME_TOP_LEFT[1] + GAME_BOARD_SIZE[1] // 2 - CUBE_SIZE // 2
    print(f"首次点击: ({initial_x}, {initial_y})")
    pyautogui.click(initial_x, initial_y)
    time.sleep(1.5)

    # 主循环
    while True:
        # 1. 截图
        image_data_url = screenshot_to_data_url(GAME_REGION)

        # 2. 异步调用AI分析
        agent_response = await get_safe_move_from_ai(model, image_data_url)

        # 3. 解析结果（核心修复：适配AIMessage对象）
        safe_cell = parse_agent_response(agent_response)
        if safe_cell:
            print(f"AI 决定点击安全格子: {safe_cell}")
            screen_x, screen_y = grid_to_screen_coords(safe_cell)
            print(f"转换为屏幕坐标: ({screen_x:.2f}, {screen_y:.2f})")
            pyautogui.moveTo(screen_x, screen_y, duration=0.2)
            pyautogui.click()
            time.sleep(1.0)
        else:
            print("AI 未找到安全格子，或游戏已结束。任务终止。")
            break


if __name__ == "__main__":
    # 兼容Windows异步事件循环
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
        else:
            raise
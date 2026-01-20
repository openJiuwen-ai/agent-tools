import os
import sys
import math
import time
import json
import base64
import pyautogui
import traceback
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

# 模拟滚动操作的默认幅度
SCROLL_AMOUNTS = {
    "small": 50,
    "medium": 200,
    "large": 500,
}

def capture_screenshot(save_path="screenshot.png"):
    """
    捕获整个桌面的截图并保存到指定路径
    
    Args:
        save_path (str): 截图保存的路径，默认为 "screenshot.png"
    
    Returns:
        str: 保存的截图文件路径
    """
    try:
        # 使用pyautogui截图
        screenshot = pyautogui.screenshot()
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        # 保存截图
        screenshot.save(save_path)
        print(f"截图已保存到: {save_path}")
        return save_path
    except Exception as e:
        print(f"截图失败: {e}")
        # 如果截图失败，尝试使用备用方法
        try:
            # 备用方法：使用PIL的ImageGrab（在Windows上可能更可靠）
            from PIL import ImageGrab
            screenshot = ImageGrab.grab()
            screenshot.save(save_path)
            print(f"使用备用方法截图成功: {save_path}")
            return save_path
        except Exception as e2:
            print(f"备用截图方法也失败: {e2}")
            raise Exception("无法捕获桌面截图")

# 图片编码函数
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded_string}"

def get_response(instruction, vl_high_resolution_images, min_pixels, max_pixels, system_prompt, 
                 screenshot_path="screenshot.png", capture_new=True):
    """
    获取GUI模型的响应
    
    Args:
        instruction (str): 用户指令
        vl_high_resolution_images (bool): 是否使用高分辨率图像
        min_pixels (int): 最小像素数
        max_pixels (int): 最大像素数
        system_prompt (str): 系统提示词
        screenshot_path (str): 截图保存路径，默认为 "screenshot.png"
        capture_new (bool): 是否每次调用都重新截图，默认为True
    
    Returns:
        tuple: (模型响应内容, 使用的截图路径)
    """
    try:
        # 如果capture_new为True或截图文件不存在，则重新截图
        if capture_new or not os.path.exists(screenshot_path):
            screenshot_path = capture_screenshot(screenshot_path)
        
        # 编码截图
        encoded_image_path = encode_image_to_base64(screenshot_path)
        
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": encoded_image_path},
                     "max_pixels": max_pixels,
                     "min_pixels": min_pixels},
                    {"type": "text", "text": instruction}
                ]
            },
        ]

        load_dotenv()
        client = OpenAI(
            api_key=os.getenv("GUI_PLUS_API_KEY"),
            base_url=os.getenv("DATABASE_URL"),
        )

        completion = client.chat.completions.create(
            model="gui-plus",
            messages=messages,
            extra_body={"vl_high_resolution_images": vl_high_resolution_images}
        )
        content = completion.choices[0].message.content
        return content, screenshot_path
        
    except Exception as e:
        # 获取当前函数的名称
        func_name = sys._getframe().f_code.co_name
        
        # 打印详细错误信息
        print(f"\n错误发生在函数: {func_name}")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        
        # 打印完整的错误追踪
        print("\n错误追踪:")
        traceback.print_exc()
        
        # 可以选择记录到日志文件
        with open("error_log.txt", "a", encoding="utf-8") as f:
            f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 错误发生在 {func_name}:\n")
            f.write(traceback.format_exc())
            f.write("\n" + "="*50 + "\n")
        
        raise  # 重新抛出异常，或者返回None

def parse_json(json_output):
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i + 1:])  # 删除 "```json"之前的所有内容
            json_output = json_output.split("```")[0]  # 删除 "```"之后的所有内容
            break  # 找到"```json"后退出循环
    response_dict = json.loads(json_output)
    return response_dict

def smart_size(image_path, point, factor, min_pixels, max_pixels, vl_high_resolution_images):
    """
    param
      image_path:  图像路径
      point     :  大模型返回的坐标值
      min_pixels:  输入图像的最小像素值，一般设置为默认值：4 * 28 * 28 即可
      max_pixels:  输入图像的最大像素值，超过此值则将图像的像素缩小至 max_pixels 内，与发起模型调用步骤设置的 max_pixels 应保持一致
      vl_high_resolution_images: 是否将输入图像的最大像素值提高到 16384 * 28 * 28，设置为 True 时，max_pixels的设置失效，采用固定分辨率处理图像，设置为 False 时，max_pixels 可自定义，
      与发起模型调用步骤设置的 vl_high_resolution_image 应保持一致
    return: 目标对象相对于原图的坐标
    """
    image = Image.open(image_path)

    # 获取图片的原始尺寸
    height = image.height
    width = image.width

    if vl_high_resolution_images:
        max_pixels = 16384 * 28 * 28
    else:
        max_pixels = max_pixels

    # 将高度调整为factor的整数倍
    h_bar = round(height / factor) * factor
    # 将宽度调整为factor的整数倍
    w_bar = round(width / factor) * factor
    # 对图像进行缩放处理，调整像素的总数在范围[min_pixels,max_pixels]内
    if h_bar * w_bar > max_pixels:
        # 计算缩放因子beta，使得缩放后的图像总像素数不超过max_pixels
        beta = math.sqrt((height * width) / max_pixels)
        # 重新计算调整后的高度，确保为factor的整数倍
        h_bar = math.floor(height / beta / factor) * factor
        # 重新计算调整后的宽度，确保为factor的整数倍
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        # 计算缩放因子beta，使得缩放后的图像总像素数不低于min_pixels
        beta = math.sqrt(min_pixels / (height * width))
        # 重新计算调整后的高度，确保为factor的整数倍
        h_bar = math.ceil(height * beta / factor) * factor
        # 重新计算调整后的宽度，确保为factor的整数倍
        w_bar = math.ceil(width * beta / factor) * factor
    abs_x1 = int(point["x"] / w_bar * width)
    abs_y1 = int(point["y"] / h_bar * height)
    return abs_x1, abs_y1

def execute_gui_action(gui_action, gui_parameters, mapped_x = 0, mapped_y = 0):
    """
    根据模型输出的动作和参数执行GUI操作。

    Args:
        gui_action (str): 模型输出的动作类型（如 "CLICK", "TYPE"）。
        gui_parameters (dict): 动作的参数字典。
        mapped_x (int): 映射后的x坐标。
        mapped_y (int): 映射后的y坐标。
    """
    print(f"执行动作: {gui_action}, 执行动作需要的参数: {gui_parameters}")

    # 1. 点击
    if gui_action == "CLICK":
        # 确保 'x' 和 'y' 存在并为数值类型
        if "x" not in gui_parameters or "y" not in gui_parameters:
            print("错误: CLICK 动作缺少 'x' 或 'y' 坐标。")
            return

        # 基于原始屏幕执行点击事件
        try:
            pyautogui.click(mapped_x, mapped_y)
            print(f"开始执行GUI操作\n：已点击坐标 ({mapped_x}, {mapped_y})")
        except Exception as e:
            print(f"坐标映射或点击失败: {e}")
    # 2. 键入
    elif gui_action == "TYPE":
        if "text" not in gui_parameters:
            print("错误: TYPE 动作缺少 'text' 参数。")
            return

        text_to_type = gui_parameters["text"]
        needs_enter = gui_parameters.get("needs_enter", False)

        pyautogui.write(text_to_type)
        if needs_enter:
            pyautogui.press("enter")
        print(f"已输入文本: '{text_to_type}', 是否按回车: {needs_enter}")
    # 3. 滚动
    elif gui_action == "SCROLL":
        if "direction" not in gui_parameters or "amount" not in gui_parameters:
            print("错误: SCROLL 动作缺少 'direction' 或 'amount' 参数。")
            return

        direction = gui_parameters["direction"].lower()
        amount_key = gui_parameters["amount"].lower()

        scroll_value = SCROLL_AMOUNTS.get(amount_key, SCROLL_AMOUNTS["medium"])  # 默认中等

        if direction == "up":
            pyautogui.scroll(scroll_value)
            print(f"已向上滚动 {scroll_value} 单位。")
        elif direction == "down":
            pyautogui.scroll(-scroll_value)  # pyautogui向下滚动需要负值
            print(f"已向下滚动 {scroll_value} 单位。")
        else:
            print(f"警告: 未知滚动方向: {direction}")
    # 4. 按键
    elif gui_action == "KEY_PRESS":
        if "key" not in gui_parameters:
            print("错误: KEY_PRESS 动作缺少 'key' 参数。")
            return

        key_to_press = gui_parameters["key"].lower()
        pyautogui.press(key_to_press)
        print(f"已按下按键: {key_to_press}")
    # 5. 完成
    elif gui_action == "FINISH":
        message = gui_parameters.get("message", "任务已完成。")
        print(f"任务完成: {message}")
    # 6. 失败
    elif gui_action == "FAIL":
        reason = gui_parameters.get("reason", "任务失败。")
        print(f"任务失败: {reason}")
    else:
        print(f"警告: 收到未知动作类型: {gui_action}")

    # 模拟人类操作的延时，避免GUI操作过快
    time.sleep(1)  # 每次操作后等待1秒

def main():
    system_prompt = """
                    ## 1. 核心角色 (Core Role)
                    你是一个顶级的AI视觉操作代理。你的任务是分析电脑屏幕截图，理解用户的指令，然后将任务分解为单一、精确的GUI原子操作。

                    ## 2. [CRITICAL] JSON Schema & 绝对规则
                    你的输出**必须**是一个严格符合以下规则的JSON对象。**任何偏差都将导致失败**。

                    - **[R1] 严格的JSON**: 你的回复**必须**是且**只能是**一个JSON对象。禁止在JSON代码块前后添加任何文本、注释或解释。
                    - **[R2] 严格的Parameters结构**:`thought`对象的结构: "在这里用一句话简要描述你的思考过程。例如：用户想打开浏览器，我看到了桌面上的Chrome浏览器图标，所以下一步是点击它。"
                    - **[R3] 精确的Action值**: `action`字段的值**必须**是`## 3. 工具集`中定义的一个大写字符串（例如 `"CLICK"`, `"TYPE"`），不允许有任何前导/后置空格或大小写变化。
                    - **[R4] 严格的Parameters结构**: `parameters`对象的结构**必须**与所选Action在`## 3. 工具集`中定义的模板**完全一致**。键名、值类型都必须精确匹配。

                    ## 3. 工具集 (Available Actions)
                    ### CLICK
                    - **功能**: 单击屏幕。
                    - **Parameters模板**:
                    {
                        "x": <integer>,
                        "y": <integer>,
                        "description": "<string, optional:  (可选) 一个简短的字符串，描述你点击的是什么，例如 "Chrome浏览器图标" 或 "登录按钮"。>"
                    }
                        
                    ### TYPE
                    - **功能**: 输入文本。
                    - **Parameters模板**:
                    {
                    "text": "<string>",
                    "needs_enter": <boolean>
                    }
                        
                    ### SCROLL
                    - **功能**: 滚动窗口。
                    - **Parameters模板**:
                    {
                    "direction": "<'up' or 'down'>",
                    "amount": "<'small', 'medium', or 'large'>"
                    }

                    ### KEY_PRESS
                    - **功能**: 按下功能键。
                    - **Parameters模板**:
                    {
                    "key": "<string: e.g., 'enter', 'esc', 'alt+f4'>"
                    }
                        
                    ### FINISH
                    - **功能**: 任务成功完成。
                    - **Parameters模板**:
                    {
                    "message": "<string: 总结任务完成情况>"
                    }
                        
                    ### FAIL
                    - **功能**: 任务无法完成。
                    - **Parameters模板**:
                    {
                    "reason": "<string: 清晰解释失败原因>"
                    }

                    ## 4. 思维与决策框架
                    在生成每一步操作前，请严格遵循以下思考-验证流程：

                    目标分析: 用户的最终目标是什么？
                    屏幕观察 (Grounded Observation): 仔细分析截图。你的决策必须基于截图中存在的视觉证据。 如果你看不见某个元素，你就不能与它交互。
                    行动决策: 基于目标和可见的元素，选择最合适的工具。
                    构建输出:
                    a. 在thought字段中记录你的思考。
                    b. 选择一个action。
                    c. 精确复制该action的parameters模板，并填充值。
                    最终验证 (Self-Correction): 在输出前，最后检查一遍：
                    我的回复是纯粹的JSON吗？
                    action的值是否正确无误（大写、无空格）？
                    parameters的结构是否与模板100%一致？例如，对于CLICK，是否有独立的x和y键，并且它们的值都是整数？
                    """
    
    # instruction = "打开Edge浏览器"
    # instruction = "输入: Hello World!不要回车"
    instruction = "向下滚动页面"
    # instruction = "按键K"
    # instruction = "点击屏幕上方的搜索框"
    
    min_pixels = 4 * 28 * 28
    max_pixels = 1280 * 28 * 28
    factor = 28
    vl_high_resolution_images = True

    try:
        # 调用get_response，自动截图
        model_response, screenshot_path = get_response(
            instruction=instruction,
            vl_high_resolution_images=vl_high_resolution_images,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            system_prompt=system_prompt,
            screenshot_path="screenshot_logs/current_screenshot.png",  # 可以自定义保存路径
            capture_new=True  # 每次调用都重新截图
        )

        print(f"使用的截图路径: {screenshot_path}")
        print("大模型的回复：", model_response)
    
        response_dict = parse_json(model_response)
        action = response_dict["action"]
        # 去除潜在的空格
        action = action.replace(" ", "")

        # 点击操作需要归一化分辨率
        if action == "CLICK":
            parameters = response_dict["parameters"]
            abs_x1, abs_y1 = smart_size(screenshot_path,
                                        parameters,
                                        factor=factor,
                                        min_pixels=min_pixels,
                                        max_pixels=max_pixels,
                                        vl_high_resolution_images=vl_high_resolution_images)
            # 只有在所有变量都成功定义后才执行GUI操作
            execute_gui_action(action, parameters, abs_x1, abs_y1)
        else:
            execute_gui_action(action, response_dict)

    except Exception as e:
        print(f"\n主程序执行失败:")
        print(f"错误位置: {traceback.extract_tb(sys.exc_info()[2])[-1]}")
        print(f"详细信息:")
        traceback.print_exc()
        
        # 更详细的错误位置信息
        exc_type, exc_value, exc_traceback = sys.exc_info()
        
        # 获取最后一个错误的帧信息
        last_frame = traceback.extract_tb(exc_traceback)[-1]
        print(f"\n📍 具体位置:")
        print(f"  文件: {last_frame.filename}")
        print(f"  行号: {last_frame.lineno}")
        print(f"  函数: {last_frame.name}")
        print(f"  代码: {last_frame.line}")

        
if __name__ == "__main__":
    main()
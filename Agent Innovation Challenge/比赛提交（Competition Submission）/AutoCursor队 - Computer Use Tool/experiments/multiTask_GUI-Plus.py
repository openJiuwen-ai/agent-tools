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
    """解析模型返回的JSON字符串"""
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
    将模型返回的坐标转换为原始屏幕坐标
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

def execute_gui_action(gui_action, gui_parameters, mapped_x=0, mapped_y=0):
    """
    根据模型输出的动作和参数执行GUI操作。
    """
    print(f"执行动作: {gui_action}, 执行参数: {gui_parameters}")
    
    try:
        # 1. 点击
        if gui_action == "CLICK":
            # 确保 'x' 和 'y' 存在并为数值类型
            if "x" not in gui_parameters or "y" not in gui_parameters:
                print("错误: CLICK 动作缺少 'x' 或 'y' 坐标。")
                return False

            # 基于原始屏幕执行点击事件
            try:
                pyautogui.click(mapped_x, mapped_y)
                print(f"已单击坐标 ({mapped_x}, {mapped_y})")
                return True
            except Exception as e:
                print(f"坐标映射或单击失败: {e}")
                return False
        
        # 双击
        elif gui_action == "DOUBLECLICK":
            # 确保 'x' 和 'y' 存在并为数值类型
            if "x" not in gui_parameters or "y" not in gui_parameters:
                print("错误: DOUBLECLICK 动作缺少 'x' 或 'y' 坐标。")
                return False

            # 基于原始屏幕执行点击事件
            try:
                pyautogui.doubleClick(mapped_x, mapped_y)
                print(f"已双击坐标 ({mapped_x}, {mapped_y})")
                return True
            except Exception as e:
                print(f"坐标映射或双击失败: {e}")
                return False

        # 2. 键入
        elif gui_action == "TYPE":
            if "text" not in gui_parameters:
                print("错误: TYPE 动作缺少 'text' 参数。")
                return False

            text_to_type = gui_parameters["text"]
            needs_enter = gui_parameters.get("needs_enter", True)

            pyautogui.write(text_to_type)
            if needs_enter:
                pyautogui.press("enter")
            print(f"已输入文本: '{text_to_type}', 是否按回车: {needs_enter}")
            return True
        
        # 3. 滚动
        elif gui_action == "SCROLL":
            if "direction" not in gui_parameters or "amount" not in gui_parameters:
                print("错误: SCROLL 动作缺少 'direction' 或 'amount' 参数。")
                return False

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
                return False
            return True
        
        # 4. 按键
        elif gui_action == "KEY_PRESS":
            if "key" not in gui_parameters:
                print("错误: KEY_PRESS 动作缺少 'key' 参数。")
                return False

            key_to_press = gui_parameters["key"].lower()
            pyautogui.press(key_to_press)
            print(f"已按下按键: {key_to_press}")
            return True
        
        # 5. 完成
        elif gui_action == "FINISH":
            message = gui_parameters.get("message", "任务已完成。")
            print(f"任务完成: {message}")
            return True
        
        # 6. 失败
        elif gui_action == "FAIL":
            reason = gui_parameters.get("reason", "任务失败。")
            print(f"任务失败: {reason}")
            return False
        
        else:
            print(f"警告: 收到未知动作类型: {gui_action}")
            return False

    except Exception as e:
        print(f"执行动作时发生异常: {e}")
        traceback.print_exc()
        return False
    
    finally:
        # 模拟人类操作的延时，避免GUI操作过快
        time.sleep(1)  # 每次操作后等待1秒

def parse_complex_instruction(instruction, max_steps=10):
    """
    使用大模型解析复杂指令为多个单步指令
    
    Args:
        instruction (str): 复杂指令
        max_steps (int): 最大步骤数
    
    Returns:
        list: 单步指令列表
    """
    system_prompt = """
                    你是一个任务分解专家。你的任务是将用户的复杂GUI操作指令分解为多个原子步骤。
                    输出格式要求：
                    你必须严格输出以下JSON格式，不能有任何额外的文本、注释或解释：
                    ```json
                    {
                        "thought": "简要分析用户指令，说明如何分解",
                        "steps": [
                            "第一步的具体指令",
                            "第二步的具体指令",
                            "第三步的具体指令"
                        ]
                    }
                    ```
                    
                    示例：
                    输入："在哔哩哔哩搜索huawei"
                    
                    输出：
                    ```json
                    {
                        "thought": "用户指令包含三个原子操作：1.点击链接，2.点击搜索框，3.输入文本并回车",
                        "steps": [
                            "点击哔哩哔哩链接",
                            "点击搜索框",
                            "输入:huawei并回车"
                        ]
                    }
                    ```
                    
                    规则：
                    1. 每个步骤应该是原子操作，可以在一次截图分析中完成
                    2. 保持原指令的语义不变
                    3. 步骤数量不超过10个
                    4. 如果已经是单步指令，直接放入steps数组
                    5. 如果用户的原始指令中包含"打开","双击",在分解指令时要明确为"双击"
                    """
    
    # 模型参数
    min_pixels = 4 * 28 * 28
    max_pixels = 1280 * 28 * 28
    vl_high_resolution_images = True
    
    try:
        # 先截图当前状态
        screenshot_path = capture_screenshot("screenshot_logs/instruction_parse.png")
        
        # 调用模型分解指令
        model_response, _ = get_response(
            instruction=f"请将以下复杂指令分解为多个单步GUI操作：{instruction}",
            vl_high_resolution_images=vl_high_resolution_images,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            system_prompt=system_prompt,
            screenshot_path=screenshot_path,
            capture_new=False  # 使用刚截的图
        )
        
        print(f"指令分解响应: {model_response}")
        
        # 解析JSON响应
        response_dict = parse_json(model_response)
        steps = response_dict.get("steps", [])
        thought = response_dict.get("thought", "无分析")
        
        print(f"指令分解分析: {thought}")
        print(f"分解出 {len(steps)} 个步骤: {steps}")
        
        return steps
        
    except Exception as e:
        print(f"指令分解失败: {e}")
        # 如果分解失败，使用简单的分割方法
        print("使用备用方法分解指令...")
        
        # 简单的分割方法
        steps = []
        # 按中文分隔符分割
        if "，" in instruction:
            raw_steps = instruction.split("，")
        elif "并" in instruction:
            raw_steps = instruction.split("并")
        elif "然后" in instruction:
            raw_steps = instruction.split("然后")
        else:
            # 单个指令
            raw_steps = [instruction]
        
        # 清理每个步骤
        for step in raw_steps:
            step = step.strip()
            if step:
                # 处理"输入:"前缀
                if step.startswith("输入:"):
                    step = step.replace("输入:", "输入: ")
                steps.append(step)
        
        print(f"备用方法分解出 {len(steps)} 个步骤: {steps}")
        return steps

def execute_single_step(step_instruction, step_num, total_steps):
    """
    执行单个步骤
    
    Args:
        step_instruction (str): 单步指令
        step_num (int): 当前步骤序号
        total_steps (int): 总步骤数
    
    Returns:
        bool: 是否执行成功
    """
    print(f"\n{'='*60}")
    print(f"步骤 {step_num}/{total_steps}: {step_instruction}")
    print(f"{'='*60}")
    
    system_prompt = """
                    ## 1. 核心角色 (Core Role)
                    你是一个顶级的AI视觉操作代理。你的任务是分析电脑屏幕截图，理解用户的指令，然后执行单一的GUI原子操作。

                    ## 2. [CRITICAL] JSON Schema & 绝对规则
                    你的输出**必须**是一个严格符合以下规则的JSON对象。**任何偏差都将导致失败**。

                    - **[R1] 严格的JSON**: 你的回复**必须**是且**只能是**一个JSON对象。禁止在JSON代码块前后添加任何文本、注释或解释。
                    - **[R2] 严格的`thought`对象的结构**: "在这里用一句话简要描述你的思考过程。例如：用户想打开浏览器，我看到了桌面上的Chrome浏览器图标，所以下一步是点击它。"
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
                    - **返回示例**：
                    ```json
                    {
                        "action": " CLICK",
                        "parameters": {
                            "x": 907,
                            "y": 185,
                            "description": "点击浏览器按钮"
                        }
                    }
                    ```

                    ### DOUBLECLICK
                    - **功能**: 双击屏幕。
                    - **Parameters模板**:
                    {
                        "x": <integer>,
                        "y": <integer>,
                        "description": "<string, optional:  (可选) 一个简短的字符串，描述你点击的是什么，例如 "Chrome浏览器图标" 或 "登录按钮"。>"
                    }
                    - **返回示例**：
                    ```json
                    {
                        "action": " DOUBLECLICK",
                        "parameters": {
                            "x": 907,
                            "y": 185,
                            "description": "双击浏览器图标"
                        }
                    }
                    ```

                    ### TYPE
                    - **功能**: 输入文本。
                    - **Parameters模板**:
                    {
                        "text": "<string>",
                        "needs_enter": <boolean>
                    }
                    - **返回示例**：
                    ```json
                    {
                        "action": " TYPE",
                        "parameters": {
                            "text": "huawei"
                        }
                    }
                    ```
                        
                    ### SCROLL
                    - **功能**: 滚动窗口。
                    - **Parameters模板**:
                    {
                        "direction": "<'up' or 'down'>",
                        "amount": "<'small', 'medium', or 'large'>"
                    }
                    - **返回示例**：
                    ```json
                    {
                        "action": " SCROLL",
                        "parameters": {
                            "direction": "down",
                            "amount": "medium"
                        }
                    }
                    ```

                    ### KEY_PRESS
                    - **功能**: 按下功能键。
                    - **Parameters模板**:
                    {
                        "key": "<string: e.g., 'enter', 'esc', 'alt+f4'>"
                    }
                    - **返回示例**：
                    ```json
                    {
                        "action": " KEY_PRESS",
                        "parameters": {
                            "key": "enter",
                        }
                    }
                    ```
                        
                    ### FINISH
                    - **功能**: 当前步骤成功完成。
                    - **Parameters模板**:
                    {
                        "message": "<string: 完成消息>"
                    }
                        
                    ### FAIL
                    - **功能**: 当前步骤无法完成。
                    - **Parameters模板**:
                    {
                        "reason": "<string: 失败原因>"
                    }

                    ## 4. 重要说明
                    你只需要完成当前步骤的指令，不需要考虑后续步骤。
                    当前步骤完成后，请使用FINISH动作。
                    如果当前步骤无法完成，请使用FAIL动作。
                    构建输出: 精确复制该action的parameters模板，并填充值。
                    校验parameters的结构是否与模板100%一致？例如，对于CLICK，是否有独立的x和y键，并且它们的值都是整数？
                    如操作中包含"点击"，使用CLICK, 如包含"打开"，使用DOUBLECLICK。
                    """
    
    # 模型参数
    min_pixels = 4 * 28 * 28
    max_pixels = 1280 * 28 * 28
    factor = 28
    vl_high_resolution_images = True
    
    try:
        # 为每个步骤创建独立的截图文件
        screenshot_dir = "screenshot_logs"
        os.makedirs(screenshot_dir, exist_ok=True)
        screenshot_path = f"{screenshot_dir}/step_{step_num}.png"
        
        # 调用模型获取当前步骤的响应
        model_response, screenshot_path = get_response(
            instruction=step_instruction,
            vl_high_resolution_images=vl_high_resolution_images,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            system_prompt=system_prompt,
            screenshot_path=screenshot_path,
            capture_new=True
        )
        
        print(f"使用的截图路径: {screenshot_path}")
        print(f"模型响应: {model_response}")
        
        # 解析JSON响应
        response_dict = parse_json(model_response)
        action = response_dict.get("action", "").strip().replace(" ", "")
        parameters = response_dict.get("parameters", {})
        thought = response_dict.get("thought", "无思考过程")
        
        print(f"思考过程: {thought}")
        print(f"解析出的动作: {action}")
        
        # 如果是FAIL动作，返回失败
        if action == "FAIL":
            reason = parameters.get("reason", "步骤执行失败")
            print(f"❌ 步骤 {step_num} 失败: {reason}")
            return False
        
        # 坐标映射（如果需要）
        mapped_x, mapped_y = 0, 0
        if action == "CLICK" or action == "DOUBLECLICK":
            mapped_x, mapped_y = smart_size(
                screenshot_path,
                parameters,
                factor=factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                vl_high_resolution_images=vl_high_resolution_images
            )
            print(f"坐标映射: ({parameters.get('x')}, {parameters.get('y')}) -> ({mapped_x}, {mapped_y})")
        
        # 执行动作
        success = execute_gui_action(action, parameters, mapped_x, mapped_y)

        if success:
            print(f"✅ 步骤 {step_num} 执行成功")
            
            # 如果不是最后一步，等待界面响应
            if step_num < total_steps:
                print(f"⏳ 等待界面响应...")
                time.sleep(2)  # 步骤间等待
            
            return True
        else:
            print(f"❌ 步骤 {step_num} 执行失败")
            return False
            
    except Exception as e:
        print(f"\n步骤 {step_num} 执行时发生错误:")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        traceback.print_exc()
        
        # 记录错误日志
        with open("error_log.txt", "a", encoding="utf-8") as f:
            f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 步骤 {step_num} 错误:\n")
            f.write(traceback.format_exc())
            f.write("\n" + "="*50 + "\n")
        
        return False

def run_multistep_task(initial_instruction, max_steps=10):
    """
    执行多步骤任务
    
    Args:
        initial_instruction (str): 初始复杂指令
        max_steps (int): 最大执行步骤数
    
    Returns:
        bool: 任务是否成功完成
    """
    print(f"\n🎯 开始执行多步骤任务")
    print(f"初始指令: {initial_instruction}")
    print(f"{'='*60}")
    
    # 第一步：解析复杂指令为多个单步
    print("\n📝 正在解析复杂指令...")
    step_instructions = parse_complex_instruction(initial_instruction, max_steps)
    
    if not step_instructions:
        print("❌ 无法解析指令")
        return False
    
    # 限制步骤数量
    if len(step_instructions) > max_steps:
        print(f"⚠️ 步骤数量({len(step_instructions)})超过限制({max_steps})，将截断")
        step_instructions = step_instructions[:max_steps]
    
    print(f"\n📋 将执行 {len(step_instructions)} 个步骤:")
    for i, step in enumerate(step_instructions, 1):
        print(f"  {i}. {step}")
    
    # 第二步：按顺序执行每个步骤
    print(f"\n🚀 开始执行步骤...")
    
    for i, step_instruction in enumerate(step_instructions, 1):
        success = execute_single_step(step_instruction, i, len(step_instructions))
        
        if not success:
            print(f"\n❌ 任务失败于第 {i} 步")
            return False
    
    print(f"\n{'='*60}")
    print(f"✅ 所有步骤执行完成！")
    print(f"共执行 {len(step_instructions)} 个步骤")
    print(f"{'='*60}")
    
    return True

def main():
    """支持交互式多步骤任务执行"""
    print("🤖 GUI多步骤操作代理")
    print("="*60)
    
    # 配置选项
    print("\n请选择模式:")
    print("1. 测试模式（使用预设指令）")
    print("2. 交互模式（输入自定义指令）")
    
    mode_choice = input("\n请输入模式编号 (1/2，默认1): ").strip()
    
    if mode_choice == "2":
        # 交互模式
        while True:
            print(f"\n{'='*60}")
            instruction = input("\n请输入指令 (输入 'quit' 退出): ").strip()
            
            if instruction.lower() in ['quit', 'exit', 'q']:
                print("感谢使用，再见！")
                break
            
            if not instruction:
                print("指令不能为空，请重新输入")
                continue
            
            # 询问最大步骤数
            max_steps_input = input(f"最大执行步骤数 (默认10): ").strip()
            try:
                max_steps = int(max_steps_input) if max_steps_input else 10
            except ValueError:
                max_steps = 10
                print(f"输入无效，使用默认值: {max_steps}")
            
            print(f"\n开始执行指令: {instruction}")
            print(f"最大步骤数: {max_steps}")
            
            success = run_multistep_task(instruction, max_steps=max_steps)
            
            if success:
                print(f"\n✅ 指令执行成功！")
            else:
                print(f"\n❌ 指令执行失败")
            
            # 询问是否继续
            continue_choice = input("\n是否继续执行其他指令? (y/n，默认y): ").strip().lower()
            if continue_choice == 'n':
                print("感谢使用，再见！")
                break
    else:
        # 测试模式
        auto_continue = True
        # 测试指令
        test_instructions = [
            # "必应搜索huawei",
            "双击打开此电脑"
            # "打开Edge浏览器，在地址栏输入www.baidu.com，按回车键",
            # "点击开始菜单，输入记事本，按回车键",
            # "新建标签页，点击搜索框，输入huawei并回车"
        ]
        
        print(f"测试模式，将执行 {len(test_instructions)} 个测试")
        
        for i, instruction in enumerate(test_instructions, 1):
            print(f"\n{'='*60}")
            print(f"测试 {i}/{len(test_instructions)}: {instruction}")
            print(f"{'='*60}")
            
            success = run_multistep_task(instruction, max_steps=10)
            
            if success:
                print(f"\n✅ 测试 {i} 执行成功！")
            else:
                print(f"\n❌ 测试 {i} 执行失败")
            
            if i < len(test_instructions):
                input("\n按回车键继续下一个测试...")
        
        print("\n🎉 所有测试执行完毕！")

if __name__ == "__main__":
    main()
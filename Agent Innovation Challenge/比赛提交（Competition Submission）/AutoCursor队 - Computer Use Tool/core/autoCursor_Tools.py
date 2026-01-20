from openjiuwen.core.utils.tool.param import Param
from openjiuwen.core.utils.tool.tool import tool
import pyautogui
import base64
from io import BytesIO
import time
import pyperclip  # 用于操作剪贴板
import win32gui
import win32con


@tool(name="screenshot",
      description="截图插件",
      params=[
          Param(name="x", description="截图区域左上角x坐标", type="int", required=False),
          Param(name="y", description="截图区域左上角y坐标", type="int", required=False),
          Param(name="w", description="截图区域的宽", type="int", required=False),
          Param(name="h", description="截图区域的高", type="int", required=False),
          Param(name="full", description="是否全屏截图", type="bool", required=False),

      ])
def screenshot(x=0, y=0, w=0, h=0, full=False):
    if full:
        img = pyautogui.screenshot()
    else:
        img = pyautogui.screenshot(region=(x, y, w, h))
    # img.show()
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


@tool(name="dragTo",
      description="拖拉插件",
      params=[
          Param(name="x", description="拖拉目标位置x坐标", type="int", required=True),
          Param(name="y", description="拖拉目标位置y坐标", type="int", required=True),
      ])
def dragTo(x, y):
    pyautogui.dragTo(x, y, duration=0.5)
    return True


@tool(name="getPosition",
      description="获取当前鼠标的位置",
      params=[]
      )
def getPosition():
    x, y = pyautogui.position()
    return (x, y)


@tool(name="moveTo",
      description="移动鼠标到目标位置",
      params=[
          Param(name="x", description="移动目标位置x坐标", type="int", required=True),
          Param(name="y", description="移动目标位置y坐标", type="int", required=True)
      ]
      )
def moveTo(x, y):
    pyautogui.moveTo(x, y, duration=0.5)
    return True


@tool(name="findInteractionPixel",
      description="逐渐微调鼠标位置寻找可点中后交互的像素",
      params=[
          Param(name="x", description="微调起始位置x坐标", type="int", required=True),
          Param(name="y", description="微调起始位置y坐标", type="int", required=True)
      ]
      )
def findInteractionPixel(x, y):
    class CursorDetector:
        def __init__(self):
            self.last_cursor_info = None
            
        def get_cursor_info(self):
            """获取Windows系统下的光标信息"""
            try:
                # 获取当前光标句柄
                cursor_handle = win32gui.GetCursorInfo()[1]
                # 尝试获取光标类型名称
                cursor_type = "Unknown"
                
                # 常见光标类型映射
                cursor_types = {
                    win32con.IDC_ARROW: "Arrow",
                    win32con.IDC_IBEAM: "I-Beam",
                    win32con.IDC_WAIT: "Wait",
                    win32con.IDC_CROSS: "Cross",
                    win32con.IDC_UPARROW: "Up Arrow",
                    win32con.IDC_SIZE: "Size",
                    win32con.IDC_ICON: "Icon",
                    win32con.IDC_SIZENWSE: "Size NW-SE",
                    win32con.IDC_SIZENESW: "Size NE-SW",
                    win32con.IDC_SIZEWE: "Size WE",
                    win32con.IDC_SIZENS: "Size NS",
                    win32con.IDC_SIZEALL: "Size All",
                    win32con.IDC_NO: "No",
                    win32con.IDC_HAND: "Hand",
                    win32con.IDC_APPSTARTING: "App Starting",
                    win32con.IDC_HELP: "Help"
                }
                
                # 获取系统预定义光标
                for const_id, name in cursor_types.items():
                    standard_cursor = win32gui.LoadCursor(0, const_id)
                    if cursor_handle == standard_cursor:
                        cursor_type = name
                        break
                
                return (cursor_handle, cursor_type)
                
            except Exception as e:
                return (None, f"Error: {str(e)}")
        
        def detect_changes(self):        
            current_cursor = self.get_cursor_info()
            if self.last_cursor_info is None:
                # 第一次检测
                self.last_cursor_info = current_cursor
            elif current_cursor[0] != self.last_cursor_info[0]:
                # 光标发生变化
                self.last_cursor_info = current_cursor
                return True
            return False

    offset = 10
    detector = CursorDetector()   
    for i in range(-offset, offset):
        for j in range(-offset, offset):
            target_x = x + i
            target_y = y + j
            pyautogui.moveTo(target_x, target_y)
            if detector.detect_changes():
                return True
            time.sleep(0.005)

    

@tool(name="write",
      description="写入文本",
      params=[
          Param(name="text", description="需要写入的文本，可以文本", type="string", required=True),
      ]
      )
def write(text):
    # 采用复制的方法来写入文本，可以对中文进行操作
    pyperclip.copy(text)
    time.sleep(0.1)
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(0.5)
    return True


@tool(name="click",
      description="单击",
      params=[
          Param(name="x", description="目标位置x坐标", type="int", required=False),
          Param(name="y", description="目标位置y坐标", type="int", required=False),
          Param(name="button", description="使用左键还是右键", type="string", required=False)
      ]
      )
def click(x=-1, y=-1, button="left"):
    if x != -1:
        pyautogui.click(x=x, y=y, button=button, duration=0.1)  # 从移动到点击共耗时 0.1 秒（更自然）
    else:
        pyautogui.click(button=button, duration=0.1)
    return True


@tool(name="doubleClick",
      description="双击，打开图标一般需要双击",
      params=[
          Param(name="x", description="目标位置x坐标", type="int", required=False),
          Param(name="y", description="目标位置y坐标", type="int", required=False),
          Param(name="button", description="使用左键还是右键", type="string", required=False)
      ]
      )
def doubleClick(x=-1, y=-1, button="left"):
    # 采用复制的方法来写入文本，可以对中文进行操作
    if x != -1:
        pyautogui.doubleClick(x=x, y=y, button=button)  # 从移动到点击共耗时 0.1 秒（更自然）
    else:
        pyautogui.doubleClick(button=button)
    return True


@tool(name="scroll",
      description="滚动滑轮",
      params=[
          Param(name="len", description="滚动的pix", type="int", required=True)
      ]
      )
def scroll(len):
    # 采用复制的方法来写入文本，可以对中文进行操作
    pyautogui.scroll(len)  # 向上滚动 10 单位
    return True



@tool(
    name="pressEnter",
    description="按下键盘上的回车（Enter）键",
    params=[]
)
def pressEnter():
    #模拟按下 Enter 键，常用于确认输入、提交表单或执行命令。
    pyautogui.press('enter')
    return True

@tool(name="keyPress",
      description="按键",
      params=[
          Param(name="key", description="按下的按键名", type="string", required=True)
      ]
      )
def keyPress(key):
    key_to_press = key.lower()
    pyautogui.press(key_to_press)
    return True
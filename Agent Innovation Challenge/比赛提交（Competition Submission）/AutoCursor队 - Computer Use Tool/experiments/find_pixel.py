import time
import win32gui
import win32con
import pyautogui
from datetime import datetime

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
            # print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 初始光标: {current_cursor[1]}")
        elif current_cursor[0] != self.last_cursor_info[0]:
            # 光标发生变化
            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            print(f"[{timestamp}] 光标已更改: {self.last_cursor_info[1]} -> {current_cursor[1]}")
            self.last_cursor_info = current_cursor
            return True
        return False
    
def find_pixel(init_x, init_y, detector, check_interval=0.1):
    offset = 5
    for i in range(offset):
        for j in range(offset):
            target_x = init_x + i
            target_y = init_y + j
            pyautogui.moveTo(target_x, target_y)
            if detector.detect_changes():
                return target_x, target_y
            time.sleep(check_interval)

def main():
    try:
        # 创建检测器实例
        time.sleep(3)
        detector = CursorDetector()       
        init_x, init_y = 604, 193
        target_x, target_y = find_pixel(init_x, init_y, detector)
        print(target_x, target_y)
        
    except KeyboardInterrupt:
        print("\n\n检测已停止")
    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        print("\n程序结束")
    

if __name__ == "__main__":
    main()
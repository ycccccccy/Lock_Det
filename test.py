import ctypes
import time
from ctypes.wintypes import DWORD, LONG, WORD
import win32api
import win32con

# 定义 INPUT 和 MOUSEINPUT 结构体
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", LONG),
        ("dy", LONG),
        ("mouseData", DWORD),
        ("dwFlags", DWORD),
        ("time", DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
    ]

class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", DWORD),
        ("mi", MOUSEINPUT)
    ]

# 定义鼠标事件类型常量
MOUSEEVENTF_MOVE = 0x0001

# 获取屏幕分辨率
screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)

# 计算屏幕中心点坐标
screen_center_x = screen_width // 2
screen_center_y = screen_height // 2

time.sleep(1)

# 将鼠标位置设置为屏幕中心点
win32api.SetCursorPos((screen_center_x, screen_center_y))

# 输出当前鼠标位置
x, y = win32api.GetCursorPos()
print(f"移动前鼠标位置: ({x}, {y})")

time.sleep(1)
# 计算 dx 和 dy 参数的值
dx = -100
dy = 0

# 创建 INPUT 结构体实例
mi = MOUSEINPUT()
mi.dx = dx
mi.dy = dy
mi.mouseData = 0
mi.dwFlags = MOUSEEVENTF_MOVE
mi.time = 0
mi.dwExtraInfo = None

inp = INPUT()
inp.type = ctypes.c_ulong(0)
inp.mi = mi

# 使用 SendInput 函数模拟鼠标移动
ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

time.sleep(1)
# 获取鼠标移动后的位置
x, y = win32api.GetCursorPos()
print(f"移动后鼠标位置: ({x}, {y})")

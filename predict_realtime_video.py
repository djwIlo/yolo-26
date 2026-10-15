import cv2
import numpy as np
from mss import mss
import win32gui
from ultralytics import YOLO

# ========== 配置区，这里改你的参数 ==========
MODEL_PATH = r"D:\project\yolo-26\runs\detect\runs\train\yolo26_detect\weights\best.pt"
WINDOW_TITLE = "新倩女幽魂"   # 重点！改成目标窗口标题，例如"游戏"、"Notepad"
CONF_THRESHOLD = 0.5
DEVICE = "cpu"
# ===========================================

# 加载模型
model = YOLO(MODEL_PATH)

# 根据窗口标题获取窗口句柄、坐标
def get_window_region(window_name):
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd == 0:
        # 模糊搜索，窗口标题包含关键词
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if window_name in title:
                    windows.append((hwnd, title))
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        if len(windows) == 0:
            raise Exception("找不到该窗口，请核对窗口标题！")
        hwnd = windows[0][0]
    # 获取窗口坐标
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    return {"left": left, "top": top, "width": right-left, "height": bottom-top}

# 获取窗口截图区域
region = get_window_region(WINDOW_TITLE)
sct = mss()

print(f"✅ 捕获窗口区域: {region}")
print("按 q 关闭预览窗口")

while True:
    # 截取窗口画面
    screenshot = sct.grab(region)
    # mss截图转opencv格式
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # YOLO推理
    results = model(frame, conf=CONF_THRESHOLD, device=DEVICE, verbose=False)
    annotated_frame = results[0].plot()

    # 弹出实时预览窗口
    cv2.imshow("YOLO 窗口实时检测", annotated_frame)

    # 按q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
sct.close()

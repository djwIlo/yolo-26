import cv2
import numpy as np
from mss import mss
import win32gui
from ultralytics import YOLO
import serial
import time
import random

# ========== 配置区 ==========
MODEL_PATH = r"D:\project\yolo-26\runs\detect\runs\train\yolo26_detect\weights\best.pt"
WINDOW_TITLE = "新倩女幽魂"
CONF_THRESHOLD = 0.5
DEVICE = "cpu"

# 随机等待区间：每次触发后，下次等待时间在这个范围随机（单位秒）
RANDOM_MIN = 2
RANDOM_MAX = 4

# serial配置
PORT = "COM7"
BAUD = 115200

# 屏幕分辨率（用于坐标映射，和你ESP宏定义保持一致）
SCREEN_W = 1920
SCREEN_H = 1080

# HID逻辑范围（和你ESP宏定义保持不变）
X_MIN = 0
X_MAX = 4800
Y_MIN = 0
Y_MAX = 3000
# ============================

model = YOLO(MODEL_PATH)

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

def screen_to_hid(screen_x, screen_y):
    """把屏幕像素坐标，转换成ESP HID触控坐标"""
    hid_x = screen_x * X_MAX / SCREEN_W
    hid_y = screen_y * Y_MAX / SCREEN_H
    # 转成整数，因为HID上报是整数
    return round(hid_x), round(hid_y)

# 模拟正常人手按压时间
def gen_mouse_hold_ms() -> float:
    """
    生成模拟鼠标按压时长毫秒
    均值≈135，标准差25，限制 100 ~ 200ms
    大部分结果落在100‑160，越靠近两端概率越低
    """
    while True:
        # 正态分布
        val = random.normalvariate(mu=135, sigma=25)
        if 100 <= val <= 200:
            return round(val, 2)

def cmd(s):
    ser.write((s + "\n").encode())
    resp = ser.readline().decode(errors="ignore").strip()
    if resp:
        print(resp)
    time.sleep(0.02)

def tap(x, y):
    print(f"Tap at x={x}, y={y}")
    cmd(f"1 {x} {y}")      # 按下
    # time.sleep(0.08)       # 按住 80ms
    t = gen_mouse_hold_ms()
    time.sleep(t)       # 按住 80ms

    cmd("u")               # 抬起

# 根据标题获取窗口
def get_window_region(window_name):
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd == 0:
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if window_name in title:
                    windows.append((hwnd, title))
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        if len(windows) == 0:
            raise Exception("找不到窗口，请核对窗口标题")
        hwnd = windows[0][0]
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    return {"left": left, "top": top, "width": right-left, "height": bottom-top, "hwnd":hwnd}

win_info = get_window_region(WINDOW_TITLE)
region = {"left":win_info["left"], "top":win_info["top"], "width":win_info["width"], "height":win_info["height"]}
sct = mss()

# 初始化：生成第一次的随机等待时间
next_trigger_time = time.time() + random.uniform(RANDOM_MIN, RANDOM_MAX)

print(f"✅捕获窗口区域: {region}")
print(f"⏱ 触发间隔随机：{RANDOM_MIN} ~ {RANDOM_MAX}秒，按 q 关闭预览窗口")

while True:
    now = time.time()
    screenshot = sct.grab(region)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # YOLO推理
    results = model(frame, conf=CONF_THRESHOLD, device=DEVICE, verbose=False)
    res = results[0]

    # ====================== 定时触发逻辑 ======================
    if now >= next_trigger_time:
        print("\n---------- 随机时间到，读取当前帧目标 ----------")
        if res.boxes is not None and len(res.boxes) > 0:
            # 当前帧所有目标列表
            boxes_xywh = res.boxes.xywh.cpu().tolist()
            cls_ids = res.boxes.cls.cpu().tolist()
            confs = res.boxes.conf.cpu().tolist()
            all_targets = list(zip(boxes_xywh, cls_ids, confs))

            # 随机选其中1个目标
            selected = random.choice(all_targets)
            xywh, cls_id, conf = selected
            x, y, w, h = xywh
            class_name = model.names[int(cls_id)]

            # 屏幕全局中心点
            screen_x = win_info["left"] + x
            screen_y = win_info["top"] + y

            print(f"选中目标 | 类别:{class_name}, 窗口内xywh: {xywh}, 屏幕坐标({screen_x:.1f},{screen_y:.1f}), 置信度:{conf:.2f}")

            # =====================
            # 【在这里写你的操作！】
            # 例：pyautogui.moveTo / click 等
            # =====================
            # import pyautogui
            # pyautogui.click(screen_x, screen_y)
            x, y = screen_to_hid(screen_x, screen_y)
            tap(x, y)

        else:
            print("当前帧没有检测到任何目标，跳过本次操作")
        
        # 重新生成下一次的随机等待时间
        next_trigger_time = time.time() + random.uniform(RANDOM_MIN, RANDOM_MAX)
        print(f"下一次触发时间：{RANDOM_MIN}~{RANDOM_MAX}秒后")
    # =========================================================

    annotated_frame = res.plot()
    cv2.imshow("YOLO窗口实时检测", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
sct.close()
import cv2
import numpy as np
from mss import mss
import win32gui
from ultralytics import YOLO

# ========== 配置区 ==========
MODEL_PATH = r"D:\project\yolo-26\runs\detect\runs\train\yolo26_detect\weights\best.pt"
WINDOW_TITLE = "新倩女幽魂"
CONF_THRESHOLD = 0.5
DEVICE = "cpu"
# ============================

model = YOLO(MODEL_PATH)

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

print(f"✅捕获窗口区域: {region}")
print("按 q 关闭预览窗口")

while True:
    screenshot = sct.grab(region)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # YOLO推理
    results = model(frame, conf=CONF_THRESHOLD, device=DEVICE, verbose=False)
    res = results[0]

    # ============ 核心：提取类别、xywh ============
    if res.boxes is not None and len(res.boxes) >0:
        # 取出所有框的xywh、类别索引、置信度
        boxes_xywh = res.boxes.xywh.cpu().tolist()
        cls_ids = res.boxes.cls.cpu().tolist()
        confs = res.boxes.conf.cpu().tolist()

        # 循环遍历每一个检测目标
        for idx, (xywh, cls_id, conf) in enumerate(zip(boxes_xywh, cls_ids, confs)):
            x, y, w, h = xywh
            class_name = model.names[int(cls_id)]
            print(f"【检测{idx}】类别:{class_name}, xywh=[{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}], 置信度:{conf:.3f}")

            # ===== 可选：转换成【桌面屏幕全局坐标】=====
            # 窗口左上角屏幕坐标 + 图片内中心点 = 屏幕真实中心点
            screen_center_x = win_info["left"] + x
            screen_center_y = win_info["top"] + y
            print(f"    -> 屏幕全局中心点: X={screen_center_x:.1f}, Y={screen_center_y:.1f}")
    # =============================================

    annotated_frame = res.plot()
    cv2.imshow("YOLO窗口实时检测", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
sct.close()

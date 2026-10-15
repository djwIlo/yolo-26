import random
import time

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
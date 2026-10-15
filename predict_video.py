from ultralytics import YOLO

# 加载训练好的最优权重best.pt
model = YOLO(r"D:\project\yolo-26\runs\detect\runs\train\yolo26_detect\weights\best.pt")

# 视频预测，替换成你的视频文件路径
video_path = r"D:\game\倩女幽魂\录屏\倩女_01\屏幕录制 2026-10-14 171603.mp4"  # 这里改成你的视频

# 推理参数
results = model.predict(
    source=video_path,
    save=True,          # 保存画框后的视频
    save_txt=False,     # 是否保存每帧框坐标txt，不需要就False
    conf=0.5,           # 置信度阈值，低于0.5的框直接过滤，可调整0.3~0.7
    imgsz=640,
    device="cpu"        # 你是CPU推理，保持cpu；如果后续想用AMD显卡ONNX推理再改
)

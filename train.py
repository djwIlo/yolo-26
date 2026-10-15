from ultralytics import YOLO

# 加载预训练权重，可选 yolov8n(最小) / yolov8s / yolov8m
model = YOLO("yolov8s.pt")

# 开始训练
results = model.train(
    data="D:/project/yolo-26/data.yaml",
    epochs=100,
    imgsz=640,
    batch=2,          # 显存小于6G就改成4，显存很小改成2
    device="cpu",         # GPU训练写0；CPU训练改成 device="cpu"
    workers=0,        # Windows系统固定workers=0，防止报错
    project="runs/train",
    name="yolo26_detect"
)

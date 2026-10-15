from ultralytics import YOLO

# 加载最优权重
model = YOLO(r"D:\project\yolo-26\runs\detect\runs\train\yolo26_detect\weights\best.pt")

# 预测单张图片，自动保存带框的结果图
res = model.predict(r"D:\project\yolo-26\qiannv_02\image_0025.jpg", save=True)

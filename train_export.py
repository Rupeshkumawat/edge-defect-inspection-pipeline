from ultralytics import YOLO

# 1. Load baseline pre-trained model
model = YOLO("yolov8n.pt")

# 2. Export directly to ONNX format (FP32)
# dynamic=True allows variable batch sizes [batch, 3, 640, 640]
success = model.export(format="onnx", imgsz=640, dynamic=True, opset=17)
print(f"Exported successfully: {success}")

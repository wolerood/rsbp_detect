from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model("tcp://127.0.0.1:8888")
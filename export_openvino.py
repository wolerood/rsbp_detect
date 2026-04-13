#export model run once
from ultralytics import YOLO

#model = YOLO("yolo26n.pt")
model = YOLO("yolo26n_best.pt")
model.export(format="openvino", imgsz=320, half=False)
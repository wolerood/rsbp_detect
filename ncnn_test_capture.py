import cv2
from ultralytics import YOLO

img = cv2.imread("test_frame.jpg")
model = YOLO("yolo26n_ncnn_model", task="detect")
res = model(img, imgsz=320, verbose=False)
print("inference ok", len(res))
from ultralytics import YOLO
import cv2
import time

model = YOLO("yolo26n_openvino_model")
img = cv2.imread("test.jpg")

t0 = time.time()
results = model(img, imgsz=320, verbose=False)
print("Inference time:", time.time() - t0)
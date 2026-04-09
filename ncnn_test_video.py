import cv2
import time
from ultralytics import YOLO

img = cv2.imread("test_frame.jpg")
if img is None:
    raise RuntimeError("Не удалось открыть test_frame.jpg")

model = YOLO("yolo26n_ncnn_model", task="detect")

count = 0
start = time.time()

while True:
    results = model(img, imgsz=320, verbose=False)
    count += 1
    fps = count / (time.time() - start)
    print(f"\rFPS: {fps:.2f}", end="", flush=True)
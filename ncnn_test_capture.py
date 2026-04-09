import cv2
from ultralytics import YOLO

img = cv2.imread("test_frame.jpg")
if img is None:
    raise RuntimeError("Не удалось открыть test_frame.jpg")

print("shape:", img.shape)
print("dtype:", img.dtype)
print("contiguous:", img.flags["C_CONTIGUOUS"])

model = YOLO("yolo26n_ncnn_model", task="detect")
res = model(img, imgsz=320) #, verbose=False)
print("inference ok", len(res))
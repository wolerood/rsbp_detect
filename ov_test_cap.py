from ultralytics import YOLO
import cv2
import time

#export model run once
#model = YOLO("yolo26n.pt")
#model.export(format="openvino", imgsz=320, half=False)


model = YOLO("yolo26n_openvino_model")
img = cv2.imread("test.jpg")

t0 = time.time()
results = model(img, imgsz=320, verbose=False)
print("Inference time:", time.time() - t0)

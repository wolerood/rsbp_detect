import cv2
from ultralytics import YOLO
import time

cap = cv2.VideoCapture("/dev/video8", cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not cap.isOpened():
    raise RuntimeError("Не удалось открыть /dev/video8")

frame_count = 0
start_time = time.time()

# Load a YOLO26n PyTorch model
model = YOLO("yolo26n.pt")

# Export the model to NCNN format
model.export(format="ncnn")  # creates 'yolo26n_ncnn_model'

# Load the exported NCNN model
ncnn_model = YOLO("yolo26n_ncnn_model")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 0)  # переворот
    results = ncnn_model(frame, imgsz=320, verbose=False) #320
    annotated_frame = results[0].plot()

    #FPS count
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    cv2.putText(annotated_frame, f"FPS: {fps:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    cv2.imshow("USB Camera YOLO", annotated_frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import time
import numpy as np
from ultralytics import YOLO

DEVICE = "/dev/video8"
MODEL_PATH = "yolo26n_ncnn_model"
IMGSZ = 320

cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not cap.isOpened():
    raise RuntimeError(f"Не удалось открыть {DEVICE}")

model = YOLO(MODEL_PATH, task="detect")

frame_count = 0
start_time = time.time()
last_report_time = start_time

print("Старт детекции. Для остановки нажмите Ctrl+C.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Отрываем кадр от внутреннего буфера VideoCapture
        safe_frame = frame.copy()

        # Если нужен переворот, раскомментируйте следующую строку
        # safe_frame = cv2.flip(safe_frame, 0)

        # Делаем массив гарантированно непрерывным в памяти
        safe_frame = np.ascontiguousarray(safe_frame)

        results = model(safe_frame, imgsz=IMGSZ, verbose=False)

        detections = 0
        if results and len(results) > 0 and results[0].boxes is not None:
            detections = len(results[0].boxes)

        frame_count += 1
        now = time.time()
        elapsed = now - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        # Печатаем строку не на каждый кадр, а примерно 2 раза в секунду
        if now - last_report_time >= 0.5:
            print(
                f"\rFPS: {fps:.2f} | detections: {detections} | frame: {safe_frame.shape} {safe_frame.dtype}",
                end="",
                flush=True
            )
            last_report_time = now

except KeyboardInterrupt:
    print("\nОстановлено пользователем.")

finally:
    cap.release()
    print("Камера освобождена.")
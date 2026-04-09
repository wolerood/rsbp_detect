import cv2

cap = cv2.VideoCapture("/dev/video8", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not cap.isOpened():
    raise RuntimeError("Не удалось открыть /dev/video8")

ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Не удалось получить кадр")

print("shape:", frame.shape)
print("dtype:", frame.dtype)
print("contiguous:", frame.flags["C_CONTIGUOUS"])

cv2.imwrite("test_frame.jpg", frame)
print("saved: test_frame.jpg")
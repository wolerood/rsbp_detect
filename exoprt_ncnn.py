# запустить  один раз для экспорта модели
# модель будет создана в ./best_ncnn_model

from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="ncnn")
import os
import random
import shutil
from pathlib import Path

# ===== НАСТРОЙКИ =====
SOURCE_DIR = r"c:\My\MyProject\OKF\53 ОКФ модерн. СУ вибро\software\DataSet_red_sweetie\new"          # исходная папка
OUTPUT_DIR = r"c:\My\MyProject\OKF\53 ОКФ модерн. СУ вибро\software\DataSet_red_sweetie\new_dataset"       # куда сохранять
TRAIN_RATIO = 0.8            # доля train
SEED = 42                    # фиксируем случайность
MOVE_FILES = False           # True = перемещать, False = копировать

# =====================

random.seed(SEED)

source_path = Path(SOURCE_DIR)
output_path = Path(OUTPUT_DIR)

train_path = output_path / "train"
test_path = output_path / "test"

train_path.mkdir(parents=True, exist_ok=True)
test_path.mkdir(parents=True, exist_ok=True)

def process_folder(folder: Path):
    files = [f for f in folder.iterdir() if f.is_file()]
    if not files:
        return

    random.shuffle(files)

    split_idx = int(len(files) * TRAIN_RATIO)
    train_files = files[:split_idx]
    test_files = files[split_idx:]

    relative = folder.relative_to(source_path)

    train_subdir = train_path / relative
    test_subdir = test_path / relative

    train_subdir.mkdir(parents=True, exist_ok=True)
    test_subdir.mkdir(parents=True, exist_ok=True)

    for f in train_files:
        dest = train_subdir / f.name
        if MOVE_FILES:
            shutil.move(str(f), str(dest))
        else:
            shutil.copy2(str(f), str(dest))

    for f in test_files:
        dest = test_subdir / f.name
        if MOVE_FILES:
            shutil.move(str(f), str(dest))
        else:
            shutil.copy2(str(f), str(dest))

# обход всех подпапок
for root, dirs, files in os.walk(source_path):
    process_folder(Path(root))

print("Разделение завершено.")
# config.py
import torch
from pathlib import Path


class Config:
    # Пути
    DATA_DIR = Path("data/classification")
    MODEL_DIR = Path("models")
    RESULTS_DIR = Path("results")

    # Классы
    CLASS_NAMES = ['no_defect', 'crack', 'dent', 'scratch']
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

    # Параметры обучения
    BATCH_SIZE = 32
    IMG_SIZE = 224
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    NUM_CLASSES = len(CLASS_NAMES)

    # Устройство
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Создание директорий
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


config = Config()
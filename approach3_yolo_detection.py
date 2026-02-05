# approach3_yolo_simple.py
# В начало любого скрипта добавьте:
import matplotlib
matplotlib.use('Agg')  # Использует non-interactive backend
import matplotlib.pyplot as plt

"""
Упрощенный подход 3: Детекция объектов с YOLO
"""
import os
import sys
from pathlib import Path
import shutil
import yaml
import numpy as np
import pandas as pd
import cv2
import json
import time
from tqdm import tqdm

# Проверяем наличие ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    print("Установите ultralytics: pip install ultralytics")
    sys.exit(1)


def prepare_yolo_data_simple():
    """Упрощенная подготовка данных для YOLO"""
    print("\nПодготовка данных для YOLO...")

    # Создаем структуру
    yolo_dir = Path("data/detection_yolo")
    yolo_dir.mkdir(parents=True, exist_ok=True)

    (yolo_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Классы для детекции
    defect_classes = ['crack', 'dent', 'scratch']
    class_to_id = {name: idx for idx, name in enumerate(defect_classes)}

    # Используем те же данные, что и для классификации
    source_dirs = {
        'train': Path("data/classification/train"),
        'val': Path("data/classification/val")
    }

    for split, source_dir in source_dirs.items():
        print(f"\nОбработка {split} данных...")

        for class_name in defect_classes:
            class_dir = source_dir / class_name
            if not class_dir.exists():
                continue

            images = list(class_dir.glob('*.jpg'))
            print(f"  {class_name}: {len(images)} изображений")

            for img_path in tqdm(images, desc=f"  {class_name}"):
                # Копируем изображение
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                h, w = img.shape[:2]
                dst_img_path = yolo_dir / "images" / split / img_path.name
                cv2.imwrite(str(dst_img_path), img)

                # Создаем искусственный bounding box в центре
                class_id = class_to_id[class_name]

                # Размер бокса - 40-60% от изображения
                box_w = w * np.random.uniform(0.4, 0.6)
                box_h = h * np.random.uniform(0.4, 0.6)

                # Центр
                center_x = w * 0.5 + np.random.uniform(-0.1, 0.1) * w
                center_y = h * 0.5 + np.random.uniform(-0.1, 0.1) * h

                # Конвертируем в YOLO формат
                x_center = center_x / w
                y_center = center_y / h
                width = box_w / w
                height = box_h / h

                # Создаем файл с разметкой
                label_path = yolo_dir / "labels" / split / f"{img_path.stem}.txt"
                with open(label_path, 'w') as f:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Создаем data.yaml
    yaml_content = f"""
path: {yolo_dir.absolute()}
train: images/train
val: images/val

names:
  0: crack
  1: dent
  2: scratch
"""

    yaml_path = yolo_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n✅ Данные подготовлены: {yaml_path}")
    print(f"   Train: {len(list((yolo_dir / 'images' / 'train').glob('*.jpg')))} изображений")
    print(f"   Val: {len(list((yolo_dir / 'images' / 'val').glob('*.jpg')))} изображений")

    return str(yaml_path)


def train_yolo_simple(yaml_path, epochs=30):
    """Упрощенное обучение YOLO"""
    print("\n" + "=" * 60)
    print("ПОДХОД 3: ДЕТЕКЦИЯ ОБЪЕКТОВ (YOLOv8n)")
    print("=" * 60)

    start_time = time.time()

    # Загружаем предобученную модель
    print("Загрузка YOLOv8n...")
    model = YOLO('yolov8n.pt')

    # Обучаем модель (упрощенный вариант)
    print(f"\nОбучение YOLO на {epochs} эпох...")

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=8,
        patience=5,
        save=True,
        project='yolo_training',
        name='defect_detection_simple',
        verbose=False  # Убираем подробный вывод
    )

    training_time = time.time() - start_time
    print(f"\n✅ Обучение завершено за {training_time:.2f} секунд")

    # Находим лучшую модель
    model_path = Path('yolo_training/defect_detection_simple/weights/best.pt')
    if model_path.exists():
        print(f"✅ Лучшая модель сохранена: {model_path}")
        return str(model_path), training_time
    else:
        print("⚠️  Лучшая модель не найдена, используем последнюю")
        return 'yolo_training/defect_detection_simple/weights/last.pt', training_time


def evaluate_yolo_simple(model_path, test_dir="data/classification/test"):
    """Упрощенная оценка YOLO"""
    print("\n🔍 Оценка YOLO на тестовых данных...")

    # Загружаем модель
    model = YOLO(model_path)

    # Тестовые изображения
    test_images = []
    for class_name in ['crack', 'dent', 'scratch']:
        class_dir = Path(test_dir) / class_name
        if class_dir.exists():
            test_images.extend(list(class_dir.glob('*.jpg'))[:10])  # по 10 каждого класса

    if not test_images:
        print("⚠️  Нет тестовых изображений!")
        return {}

    print(f"Тестирование на {len(test_images)} изображениях...")

    all_results = []
    detection_counts = {0: 0, 1: 0, 2: 0}  # crack, dent, scratch

    for img_path in tqdm(test_images, desc="Детекция"):
        # Определяем истинный класс из пути
        true_class = img_path.parent.name
        class_to_id = {'crack': 0, 'dent': 1, 'scratch': 2}
        true_id = class_to_id.get(true_class, -1)

        # Предсказание
        results = model(str(img_path), conf=0.25, iou=0.5, verbose=False)

        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()

                for box, conf, cls in zip(boxes, confs, classes):
                    class_id = int(cls)
                    detection_counts[class_id] = detection_counts.get(class_id, 0) + 1

                    all_results.append({
                        'image': img_path.name,
                        'true_class': true_class,
                        'detected_class': ['crack', 'dent', 'scratch'][class_id],
                        'confidence': float(conf),
                        'bbox': box.tolist()
                    })

    # Статистика
    total_detections = sum(detection_counts.values())

    print(f"\n📊 РЕЗУЛЬТАТЫ ДЕТЕКЦИИ:")
    print(f"   Всего детекций: {total_detections}")
    print(f"   Изображений обработано: {len(test_images)}")

    for class_id, count in detection_counts.items():
        class_name = ['crack', 'dent', 'scratch'][class_id]
        print(f"   {class_name}: {count} детекций")

    # Сохраняем результаты
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv("results/yolo_detections.csv", index=False)
        print(f"\n📁 Детекции сохранены в results/yolo_detections.csv")

    metrics = {
        'total_detections': total_detections,
        'images_processed': len(test_images),
        'detections_per_image': total_detections / len(test_images) if test_images else 0,
        'crack_detections': detection_counts.get(0, 0),
        'dent_detections': detection_counts.get(1, 0),
        'scratch_detections': detection_counts.get(2, 0),
    }

    return metrics


def test_yolo_speed(model_path, n_tests=20):
    """Тест скорости YOLO"""
    print("\n⚡ Тестирование скорости YOLO...")

    model = YOLO(model_path)

    # Создаем тестовое изображение
    test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    test_path = "temp_test.jpg"
    cv2.imwrite(test_path, test_img)

    # Прогрев
    for _ in range(3):
        _ = model(test_path, verbose=False)

    # Тестирование
    start_time = time.time()
    for i in range(n_tests):
        _ = model(test_path, verbose=False)

    total_time = time.time() - start_time
    avg_time = total_time / n_tests
    fps = 1.0 / avg_time

    # Удаляем временный файл
    if os.path.exists(test_path):
        os.remove(test_path)

    print(f"   Время на кадр: {avg_time * 1000:.2f} мс")
    print(f"   Скорость: {fps:.2f} FPS")

    return avg_time, fps


def save_yolo_results(metrics, inference_time, fps, training_time, model_size_mb=6.2):
    """Сохранение результатов YOLO"""
    results = {
        'approach': 'Object Detection (YOLOv8n)',
        'model': 'YOLOv8n',
        'training_time': training_time,
        'inference_time_ms': inference_time * 1000,
        'fps': fps,
        'model_size_mb': model_size_mb,
        'metrics': metrics
    }

    with open("results/yolo_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📁 Результаты YOLO сохранены:")
    print("   - results/yolo_results.json")
    print("   - results/yolo_detections.csv")

    return results


def main():
    """Основная функция подхода 3"""
    print("=" * 60)
    print("ЗАПУСК ПОДХОДА 3: ДЕТЕКЦИЯ ОБЪЕКТОВ (YOLO)")
    print("=" * 60)

    # 1. Подготовка данных
    yaml_path = prepare_yolo_data_simple()

    # 2. Обучение
    model_path, training_time = train_yolo_simple(yaml_path, epochs=30)

    # 3. Оценка
    metrics = evaluate_yolo_simple(model_path)

    # 4. Тест скорости
    inference_time, fps = test_yolo_speed(model_path)

    # 5. Сохранение результатов
    results = save_yolo_results(metrics, inference_time, fps, training_time)

    print("\n" + "=" * 60)
    print("ПОДХОД 3 ЗАВЕРШЕН")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = main()
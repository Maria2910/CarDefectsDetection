# prepare_data.py
import os
import shutil
import random
from pathlib import Path
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split


def parse_xml_annotation(xml_path):
    """Парсинг XML файла для получения bounding box"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        boxes.append({
            'class': name,
            'bbox': [xmin, ymin, xmax, ymax]
        })

    return boxes


def prepare_classification_data():
    """Подготовка данных для классификации (без локализации)"""
    base_path = Path("NEU-DET")

    # Соответствие классов ТЗ
    class_mapping = {
        'crazing': 'crack',  # трещина
        'scratches': 'scratch',  # царапина/скол
        'pitted_surface': 'dent',  # вмятина (условно)
    }

    # Классы, которые будем игнорировать (не по ТЗ)
    ignore_classes = ['inclusion', 'patches', 'rolled-in_scale']

    # Создаем структуру папок
    output_dirs = {
        'train': Path("data/classification/train"),
        'val': Path("data/classification/val"),
        'test': Path("data/classification/test")
    }

    for split in ['train', 'val', 'test']:
        for class_name in ['no_defect', 'crack', 'dent', 'scratch']:
            (output_dirs[split] / class_name).mkdir(parents=True, exist_ok=True)

    # Собираем все изображения
    all_images = []
    for split in ['train', 'validation']:
        split_path = base_path / split / 'images'
        for class_folder in split_path.iterdir():
            if class_folder.is_dir():
                class_name = class_folder.name
                if class_name in ignore_classes:
                    continue  # Пропускаем ненужные классы

                label = class_mapping.get(class_name, class_name)
                for img_path in class_folder.glob('*.jpg'):
                    all_images.append({
                        'path': img_path,
                        'label': label,
                        'original_split': split
                    })

    # Разделяем на train/val/test
    train_val = [img for img in all_images if img['original_split'] == 'train']
    test = [img for img in all_images if img['original_split'] == 'validation']

    # Дополнительно разделяем train на train/val
    train, val = train_test_split(
        train_val, test_size=0.15, random_state=42,
        stratify=[img['label'] for img in train_val]
    )

    print(f"Train: {len(train)} images")
    print(f"Val: {len(val)} images")
    print(f"Test: {len(test)} images")

    # Копируем файлы
    def copy_images(images, split_name):
        for img_info in images:
            src = img_info['path']
            label = img_info['label']
            dst = output_dirs[split_name] / label / src.name
            shutil.copy2(src, dst)

    copy_images(train, 'train')
    copy_images(val, 'val')
    copy_images(test, 'test')

    # Создадим немного "нормальных" изображений (без дефектов)
    # Для этого возьмем случайные патчи из изображений с дефектами
    print("\nСоздание искусственных 'нормальных' изображений...")
    create_normal_images(train, output_dirs['train'] / 'no_defect', n_samples=100)
    create_normal_images(val, output_dirs['val'] / 'no_defect', n_samples=20)
    create_normal_images(test, output_dirs['test'] / 'no_defect', n_samples=20)

    print("Данные для классификации подготовлены!")


def create_normal_images(images, output_dir, n_samples=100):
    """Создание искусственных нормальных изображений"""
    import cv2
    import numpy as np

    if not images:
        return

    for i in range(n_samples):
        # Берем случайное изображение
        img_info = random.choice(images)
        img = cv2.imread(str(img_info['path']))

        # Вырезаем случайный патч без дефекта (просто случайная область)
        h, w = img.shape[:2]
        patch_size = 200
        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)

        patch = img[y:y + patch_size, x:x + patch_size]

        # Сохраняем
        output_path = output_dir / f"normal_{i:04d}.jpg"
        cv2.imwrite(str(output_path), patch)


def prepare_detection_data():
    """Подготовка данных для детекции (с bounding boxes)"""
    print("\nПодготовка данных для детекции...")

    base_path = Path("NEU-DET")

    # Создаем структуру в формате YOLO
    output_dir = Path("data/detection")
    (output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

    # Маппинг классов для детекции
    class_ids = {
        'crazing': 0,
        'scratches': 1,
        'pitted_surface': 2
    }

    for split in ['train', 'validation']:
        img_dir = base_path / split / 'images'
        ann_dir = base_path / split / 'annotations'

        output_split = 'train' if split == 'train' else 'val'

        for class_folder in img_dir.iterdir():
            if not class_folder.is_dir():
                continue

            class_name = class_folder.name
            if class_name not in class_ids:
                continue  # Пропускаем ненужные классы

            class_id = class_ids[class_name]

            for img_path in class_folder.glob('*.jpg'):
                # Копируем изображение
                dst_img = output_dir / 'images' / output_split / img_path.name
                shutil.copy2(img_path, dst_img)

                # Создаем YOLO разметку
                xml_path = ann_dir / f"{class_name}_{img_path.stem.split('_')[-1]}.xml"
                if xml_path.exists():
                    boxes = parse_xml_annotation(xml_path)

                    # Конвертируем в YOLO формат
                    img = cv2.imread(str(img_path))
                    h, w = img.shape[:2]

                    yolo_lines = []
                    for box in boxes:
                        if box['class'] == class_name:  # Проверяем, что класс совпадает
                            xmin, ymin, xmax, ymax = box['bbox']

                            # Конвертируем в нормализованные координаты YOLO
                            x_center = (xmin + xmax) / 2 / w
                            y_center = (ymin + ymax) / 2 / h
                            box_width = (xmax - xmin) / w
                            box_height = (ymax - ymin) / h

                            yolo_lines.append(
                                f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

                    # Сохраняем разметку
                    label_path = output_dir / 'labels' / output_split / f"{img_path.stem}.txt"
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(yolo_lines))

    # Создаем файл с названиями классов
    with open(output_dir / 'classes.txt', 'w') as f:
        f.write('\n'.join(['crack', 'scratch', 'dent']))

    print("Данные для детекции подготовлены!")


if __name__ == "__main__":
    import cv2
    import numpy as np

    print("Начинаем подготовку данных...")
    prepare_classification_data()
    prepare_detection_data()

    # Статистика
    print("\n" + "=" * 50)
    print("СТАТИСТИКА ДАННЫХ")
    print("=" * 50)

    # Для классификации
    print("\nКлассификация:")
    for split in ['train', 'val', 'test']:
        split_path = Path(f"data/classification/{split}")
        print(f"\n{split.upper()}:")
        for class_dir in split_path.iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob('*.jpg')))
                print(f"  {class_dir.name}: {count} images")

    # Для детекции
    print("\nДетекция:")
    for split in ['train', 'val']:
        img_dir = Path(f"data/detection/images/{split}")
        count = len(list(img_dir.glob('*.jpg')))
        print(f"  {split}: {count} images")

    print("\nПодготовка завершена успешно!")
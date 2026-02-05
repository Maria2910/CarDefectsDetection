# normal_from_scratches.py
import cv2
import numpy as np
import random
from pathlib import Path
import shutil
from tqdm import tqdm
import math


def find_clean_patches_from_scratches(image, scratch_bboxes, n_patches=10, min_patch_size=100):
    """
    Находит чистые (без царапин) участки на изображении с царапинами

    image: исходное изображение
    scratch_bboxes: список bounding boxes царапин [[x1, y1, x2, y2], ...]
    n_patches: сколько патчей найти
    min_patch_size: минимальный размер патча
    """
    h, w = image.shape[:2]
    clean_patches = []

    # Создаем маску царапин
    scratch_mask = np.zeros((h, w), dtype=np.uint8)
    for bbox in scratch_bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(scratch_mask, (x1, y1), (x2, y2), 255, -1)

    # Расширяем маску, чтобы исключить области ВОКРУГ царапин
    kernel = np.ones((20, 20), np.uint8)
    expanded_mask = cv2.dilate(scratch_mask, kernel, iterations=1)

    # Инвертируем маску: 255 = чистые области
    clean_mask = cv2.bitwise_not(expanded_mask)

    # Находим контуры чистых областей
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Сортируем контуры по площади (от большего к меньшему)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:20]:  # Берем самые большие области
        area = cv2.contourArea(contour)
        if area < min_patch_size * min_patch_size:
            continue

        # Получаем bounding rectangle контура
        x, y, rect_w, rect_h = cv2.boundingRect(contour)

        # Пробуем несколько случайных позиций внутри этой области
        for _ in range(5):
            if rect_w >= min_patch_size and rect_h >= min_patch_size:
                # Случайная позиция внутри контура
                patch_x = x + random.randint(0, rect_w - min_patch_size)
                patch_y = y + random.randint(0, rect_h - min_patch_size)

                # Проверяем, что патч полностью внутри чистого контура
                patch_center = (patch_x + min_patch_size // 2, patch_y + min_patch_size // 2)

                # Проверка точной принадлежности контуру
                if cv2.pointPolygonTest(contour, patch_center, False) >= 0:
                    # Проверяем, что весь патч чистый
                    patch_mask = clean_mask[patch_y:patch_y + min_patch_size,
                                 patch_x:patch_x + min_patch_size]

                    if np.all(patch_mask == 255):  # Весь патч чистый
                        patch = image[patch_y:patch_y + min_patch_size,
                                patch_x:patch_x + min_patch_size]
                        clean_patches.append(patch)

                        if len(clean_patches) >= n_patches:
                            return clean_patches

    return clean_patches


def parse_scratches_xml(xml_path):
    """Парсит XML файл с царапинами, возвращает bounding boxes"""
    import xml.etree.ElementTree as ET

    if not xml_path.exists():
        return []

    tree = ET.parse(xml_path)
    root = tree.getroot()

    bboxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == 'scratches':
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            bboxes.append([xmin, ymin, xmax, ymax])

    return bboxes


def create_normal_from_scratches_dataset():
    """
    Основная функция: создает нормальные изображения из clean участков изображений с царапинами
    """
    print("=" * 70)
    print("СОЗДАНИЕ НОРМАЛЬНЫХ ИЗОБРАЖЕНИЙ ИЗ ЧИСТЫХ УЧАСТКОВ ЦАРАПИН")
    print("=" * 70)

    # Удаляем старые нормальные изображения
    for split in ['train', 'val', 'test']:
        normal_dir = Path(f"data/classification/{split}/no_defect")
        if normal_dir.exists():
            shutil.rmtree(normal_dir)
        normal_dir.mkdir(parents=True, exist_ok=True)

    # Пути к данным
    base_path = Path("NEU-DET")

    # Собираем все изображения с царапинами
    scratches_images = []

    for split in ['train', 'validation']:
        scratches_dir = base_path / split / 'images' / 'scratches'
        annotations_dir = base_path / split / 'annotations'

        if scratches_dir.exists():
            for img_path in scratches_dir.glob('*.jpg'):
                # Находим соответствующий XML
                img_id = img_path.stem.split('_')[-1]
                xml_path = annotations_dir / f"scratches_{img_id}.xml"

                scratches_images.append({
                    'img_path': img_path,
                    'xml_path': xml_path,
                    'split': split
                })

    print(f"Найдено {len(scratches_images)} изображений с царапинами")

    # Разделяем на train/val/test
    train_scratches = [img for img in scratches_images if img['split'] == 'train']
    val_test_scratches = [img for img in scratches_images if img['split'] == 'validation']

    val_size = len(val_test_scratches) // 2
    val_scratches = val_test_scratches[:val_size]
    test_scratches = val_test_scratches[val_size:]

    print(f"\nРазделение:")
    print(f"  Train: {len(train_scratches)} изображений")
    print(f"  Val: {len(val_scratches)} изображений")
    print(f"  Test: {len(test_scratches)} изображений")

    # Создаем нормальные изображения для каждого сплита
    splits = [
        ('train', train_scratches, 150, 100, 200),  # больше патчей
        ('val', val_scratches, 30, 100, 200),
        ('test', test_scratches, 30, 100, 200)
    ]

    for split_name, scratches_list, target_count, min_size, max_size in splits:
        if not scratches_list:
            print(f"\n⚠️  Нет изображений для {split_name}!")
            continue

        print(f"\nСоздание нормальных изображений для {split_name.upper()}...")
        output_dir = Path(f"data/classification/{split_name}/no_defect")

        created_count = 0
        pbar = tqdm(total=target_count)

        # Перемешиваем список
        random.shuffle(scratches_list)

        idx = 0
        while created_count < target_count and idx < len(scratches_list):
            img_info = scratches_list[idx]
            img_path = img_info['img_path']
            xml_path = img_info['xml_path']

            # Загружаем изображение
            image = cv2.imread(str(img_path))
            if image is None:
                idx += 1
                continue

            # Парсим XML с царапинами
            scratch_bboxes = parse_scratches_xml(xml_path)

            if not scratch_bboxes:
                idx += 1
                continue

            # Находим чистые патчи
            clean_patches = find_clean_patches_from_scratches(
                image,
                scratch_bboxes,
                n_patches=5,  # 5 патчей с одного изображения
                min_patch_size=min_size
            )

            # Сохраняем найденные патчи
            for i, patch in enumerate(clean_patches):
                if created_count >= target_count:
                    break

                # Иногда ресайзим для разнообразия
                if random.random() > 0.5:
                    new_size = random.randint(min_size, max_size)
                    patch = cv2.resize(patch, (new_size, new_size))

                # Иногда поворачиваем для аугментации
                if random.random() > 0.7:
                    angle = random.choice([0, 90, 180, 270])
                    if angle == 90:
                        patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
                    elif angle == 180:
                        patch = cv2.rotate(patch, cv2.ROTATE_180)
                    elif angle == 270:
                        patch = cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Сохраняем
                output_path = output_dir / f"normal_scratch_{created_count:04d}.jpg"
                cv2.imwrite(str(output_path), patch)

                created_count += 1
                pbar.update(1)

            idx += 1

        pbar.close()

        if created_count < target_count:
            print(f"  Создано только {created_count} из {target_count}")
            print(f"  Нужно больше изображений с царапинами!")

    print("\n" + "=" * 70)
    print("СТАТИСТИКА СОЗДАННЫХ ИЗОБРАЖЕНИЙ")
    print("=" * 70)

    total_normal = 0
    for split in ['train', 'val', 'test']:
        normal_dir = Path(f"data/classification/{split}/no_defect")
        if normal_dir.exists():
            normal_count = len(list(normal_dir.glob('*.jpg')))
            total_normal += normal_count
            print(f"{split.upper()}: {normal_count} нормальных изображений")

    print(f"\nВсего создано: {total_normal} нормальных изображений")

    # Визуализация
    visualize_results()


def visualize_results():
    """Визуализация созданных нормальных изображений и их источников"""
    import matplotlib.pyplot as plt

    print("\nСоздание визуализации...")

    # Находим несколько примеров
    examples = []

    # Ищем изображение с царапинами и его нормальные патчи
    base_path = Path("NEU-DET")
    scratches_dir = base_path / 'train' / 'images' / 'scratches'

    if scratches_dir.exists():
        scratches_images = list(scratches_dir.glob('*.jpg'))[:3]

        for scratch_img_path in scratches_images:
            # Загружаем изображение с царапинами
            scratch_image = cv2.imread(str(scratch_img_path))
            scratch_image = cv2.cvtColor(scratch_image, cv2.COLOR_BGR2RGB)

            # Находим соответствующий XML
            img_id = scratch_img_path.stem.split('_')[-1]
            xml_path = base_path / 'train' / 'annotations' / f"scratches_{img_id}.xml"

            # Парсим bounding boxes
            scratch_bboxes = parse_scratches_xml(xml_path)

            # Создаем визуализацию с bounding boxes
            visualized = scratch_image.copy()
            for bbox in scratch_bboxes:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(visualized, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # Находим чистые патчи
            clean_patches = find_clean_patches_from_scratches(
                cv2.cvtColor(scratch_image, cv2.COLOR_RGB2BGR),
                scratch_bboxes,
                n_patches=3,
                min_patch_size=100
            )

            examples.append({
                'original': scratch_image,
                'visualized': visualized,
                'clean_patches': clean_patches[:3],  # первые 3 патча
                'name': scratch_img_path.name
            })

    # Создаем фигуру
    n_examples = len(examples)
    if n_examples == 0:
        print("Не найдено примеров для визуализации")
        return

    fig, axes = plt.subplots(n_examples, 5, figsize=(20, n_examples * 4))

    if n_examples == 1:
        axes = axes.reshape(1, -1)

    for row, example in enumerate(examples):
        # Оригинал с царапинами
        axes[row, 0].imshow(example['original'])
        axes[row, 0].set_title(f"Оригинал\n{example['name']}")
        axes[row, 0].axis('off')

        # С bounding boxes
        axes[row, 1].imshow(example['visualized'])
        axes[row, 1].set_title("С разметкой царапин")
        axes[row, 1].axis('off')

        # Чистые патчи
        for col in range(3):
            if col < len(example['clean_patches']):
                patch = example['clean_patches'][col]
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                axes[row, 2 + col].imshow(patch_rgb)
                axes[row, 2 + col].set_title(f"Чистый патч {col + 1}\n{patch.shape[:2]}")
                axes[row, 2 + col].axis('off')

    plt.tight_layout()
    plt.savefig("normal_from_scratches_demo.jpg", dpi=150, bbox_inches='tight')
    plt.show()

    print("Визуализация сохранена как 'normal_from_scratches_demo.jpg'")


def verify_normal_images():
    """Проверяет, что созданные изображения действительно 'нормальные'"""
    print("\n" + "=" * 70)
    print("ПРОВЕРКА КАЧЕСТВА НОРМАЛЬНЫХ ИЗОБРАЖЕНИЙ")
    print("=" * 70)

    for split in ['train', 'val', 'test']:
        normal_dir = Path(f"data/classification/{split}/no_defect")

        if not normal_dir.exists():
            print(f"\n{split.upper()}: папка не существует")
            continue

        images = list(normal_dir.glob('*.jpg'))
        if not images:
            print(f"\n{split.upper()}: нет изображений")
            continue

        print(f"\n{split.upper()}: {len(images)} изображений")

        # Проверяем несколько случайных
        sample_size = min(5, len(images))
        sample = random.sample(images, sample_size)

        for img_path in sample:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Простая проверка
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 1. Проверка на однородность (не должно быть резких переходов)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # 2. Проверка размера
            h, w = img.shape[:2]

            status = "✓ ХОРОШО" if 50 <= laplacian_var <= 5000 else "⚠️ ПОДОЗРИТЕЛЬНО"

            print(f"  {img_path.name}: {w}x{h}, "
                  f"лапласиан={laplacian_var:.1f} {status}")


def balance_dataset_if_needed():
    """
    Балансирует датасет, если нужно
    """
    print("\n" + "=" * 70)
    print("БАЛАНСИРОВКА ДАТАСЕТА")
    print("=" * 70)

    defect_counts = {}
    normal_counts = {}

    for split in ['train', 'val', 'test']:
        split_path = Path(f"data/classification/{split}")

        defect_counts[split] = {}
        normal_counts[split] = 0

        for class_dir in split_path.iterdir():
            if class_dir.is_dir():
                images = list(class_dir.glob('*.jpg'))
                count = len(images)

                if class_dir.name == 'no_defect':
                    normal_counts[split] = count
                else:
                    defect_counts[split][class_dir.name] = count

    # Вывод статистики
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}:")
        print(f"  Нормальные: {normal_counts[split]}")

        total_defects = sum(defect_counts[split].values())
        print(f"  Дефекты: {total_defects} (crack: {defect_counts[split].get('crack', 0)}, "
              f"dent: {defect_counts[split].get('dent', 0)}, "
              f"scratch: {defect_counts[split].get('scratch', 0)})")

        if total_defects > 0:
            ratio = normal_counts[split] / total_defects
            print(f"  Соотношение: {ratio:.2f}:1")

            # Рекомендации
            if ratio < 0.3:
                print("  ⚠️  Слишком мало нормальных изображений!")
                print("     Рекомендуется: Увеличить n_patches в find_clean_patches_from_scratches")
            elif ratio > 3:
                print("  ⚠️  Слишком много нормальных изображений!")
                print("     Можно уменьшить target_count для нормальных")


if __name__ == "__main__":
    try:
        import cv2
        import numpy as np
        from tqdm import tqdm
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Ошибка: {e}")
        print("Установите: pip install opencv-python numpy tqdm matplotlib")
        exit(1)

    print("=" * 70)
    print("НОРМАЛЬНЫЕ ИЗОБРАЖЕНИЯ ИЗ ЧИСТЫХ УЧАСТКОВ ЦАРАПИН")
    print("=" * 70)

    # 1. Создаем нормальные изображения
    create_normal_from_scratches_dataset()

    # 2. Проверяем качество
    verify_normal_images()

    # 3. Балансируем датасет
    balance_dataset_if_needed()

    print("\n" + "=" * 70)
    print("ГОТОВО!")
    print("=" * 70)
    print("\nНормальные изображения созданы из чистых участков изображений с царапинами.")
    print("Это самый естественный способ получить 'нормальные' изображения металла.")
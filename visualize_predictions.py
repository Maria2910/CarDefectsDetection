# visualize_predictions.py
"""
Скрипт для визуализации предсказаний модели на валидационных данных
Создает новые изображения с подписями классов
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import os


class PredictionVisualizer:
    def __init__(self, model_path="models/best_classification_model.pth"):
        """Инициализация визуализатора предсказаний"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ['no_defect', 'crack', 'dent', 'scratch']
        self.class_display_names = ['Норма', 'Трещина', 'Вмятина', 'Царапина']

        print(f"🎨 Инициализация визуализатора предсказаний...")
        print(f"   Устройство: {self.device}")
        print(f"   Классы: {self.class_display_names}")

        # Загрузка модели
        self.model = models.efficientnet_b0(pretrained=False)
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, 4)

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Модель загружена из {model_path}")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            print("   Создаем демонстрационную модель...")
            self._init_demo_model()

        self.model.to(self.device)
        self.model.eval()

        # Трансформы
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Цвета для разных классов
        self.class_colors = {
            'no_defect': (0, 255, 0),  # Зеленый - норма
            'crack': (255, 0, 0),  # Красный - трещина
            'dent': (255, 165, 0),  # Оранжевый - вмятина
            'scratch': (0, 0, 255)  # Синий - царапина
        }

    def _init_demo_model(self):
        """Инициализация демонстрационной модели (если нет сохраненной)"""

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.model.apply(init_weights)

    def predict_image(self, image_path):
        """Предсказание для одного изображения"""
        try:
            # Загрузка и преобразование
            image = Image.open(image_path).convert('RGB')
            original_size = image.size

            # Сохраняем оригинал для визуализации
            display_image = image.copy()

            # Предсказание
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

            class_idx = predicted.item()
            confidence = probabilities[0, class_idx].item()

            result = {
                'image_path': str(image_path),
                'image_name': Path(image_path).name,
                'predicted_class': self.class_names[class_idx],
                'predicted_class_display': self.class_display_names[class_idx],
                'confidence': confidence,
                'all_probabilities': probabilities.cpu().numpy()[0].tolist(),
                'original_size': original_size
            }

            return result, display_image

        except Exception as e:
            print(f"❌ Ошибка при обработке {image_path}: {e}")
            return None, None

    def add_prediction_label(self, image, prediction_result):
        """Добавление подписи с предсказанием на изображение"""
        try:
            # Создаем объект для рисования
            draw = ImageDraw.Draw(image)

            # Параметры текста
            class_name = prediction_result['predicted_class_display']
            confidence = prediction_result['confidence']
            color = self.class_colors[prediction_result['predicted_class']]

            # Формируем текст
            text = f"{class_name}: {confidence:.1%}"

            # Выбираем размер шрифта в зависимости от размера изображения
            img_width, img_height = image.size
            font_size = max(20, img_width // 30)

            try:
                # Пробуем загрузить шрифт (работает на Windows)
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                # Если шрифт не найден, используем стандартный
                font = ImageFont.load_default()

            # Позиция текста (левый верхний угол)
            text_position = (10, 10)

            # Рисуем фон для текста для лучшей читаемости
            text_bbox = draw.textbbox(text_position, text, font=font)
            padding = 5
            background_box = (
                text_bbox[0] - padding,
                text_bbox[1] - padding,
                text_bbox[2] + padding,
                text_bbox[3] + padding
            )
            draw.rectangle(background_box, fill=(0, 0, 0, 128))  # Полупрозрачный черный

            # Рисуем текст
            draw.text(text_position, text, font=font, fill=color)

            # Добавляем дополнительную информацию в правый нижний угол
            if prediction_result['confidence'] < 0.6:
                warning_text = "Низкая уверенность"
                warning_position = (img_width - 200, img_height - 40)
                draw.text(warning_position, warning_text, font=font, fill=(255, 255, 0))

            return image

        except Exception as e:
            print(f"❌ Ошибка при добавлении подписи: {e}")
            return image

    def process_validation_folder(self, input_folder="data/classification/val",
                                  output_folder="validation_predictions"):
        """Обработка всех изображений в валидационной папке"""
        print(f"\n📂 Обработка валидационных данных из: {input_folder}")

        # Создаем папку для результатов
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)

        # Создаем подпапки для разных классов
        for class_name in self.class_names:
            (output_path / class_name).mkdir(exist_ok=True)

        # Собираем статистику
        stats = {
            'total_processed': 0,
            'by_class': {cls: 0 for cls in self.class_names},
            'low_confidence': 0,
            'timestamp': datetime.now().isoformat()
        }

        # Список для хранения результатов
        all_results = []

        # Обрабатываем каждую папку с классом
        for class_idx, class_name in enumerate(self.class_names):
            class_folder = Path(input_folder) / class_name

            if not class_folder.exists():
                print(f"⚠️  Папка {class_folder} не найдена, пропускаем")
                continue

            images = list(class_folder.glob('*.jpg'))
            print(f"\n   📁 {self.class_display_names[class_idx]} ({class_name}): {len(images)} изображений")

            # Обрабатываем первые 10 изображений из каждого класса (для скорости)
            sample_size = min(10, len(images))
            for i, img_path in enumerate(images[:sample_size]):
                print(f"      Обработка {i + 1}/{sample_size}: {img_path.name}", end='\r')

                # Получаем предсказание
                result, image = self.predict_image(img_path)

                if result and image:
                    # Добавляем подпись на изображение
                    labeled_image = self.add_prediction_label(image, result)

                    # Сохраняем изображение
                    output_filename = f"{class_name}_{img_path.stem}_predicted.jpg"
                    output_filepath = output_path / class_name / output_filename
                    labeled_image.save(output_filepath, quality=95)

                    # Добавляем истинный класс в результат
                    result['true_class'] = class_name
                    result['true_class_display'] = self.class_display_names[class_idx]
                    result['correct'] = (result['predicted_class'] == class_name)

                    # Обновляем статистику
                    stats['total_processed'] += 1
                    stats['by_class'][result['predicted_class']] += 1

                    if result['confidence'] < 0.6:
                        stats['low_confidence'] += 1

                    all_results.append(result)

            print()  # Новая строка после прогресса

        # Сохраняем статистику
        self._save_statistics(stats, all_results, output_path)

        print(f"\n✅ Обработка завершена!")
        print(f"   📊 Обработано изображений: {stats['total_processed']}")
        print(f"   💾 Результаты сохранены в: {output_folder}")

        return all_results, stats

    def _save_statistics(self, stats, results, output_path):
        """Сохранение статистики и результатов"""
        # JSON файл с детальными результатами
        detailed_results = {
            'metadata': {
                'timestamp': stats['timestamp'],
                'total_images': stats['total_processed'],
                'model_used': 'EfficientNet-B0'
            },
            'statistics': stats,
            'predictions': results
        }

        with open(output_path / "predictions_detailed.json", 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        # Текстовый отчет
        report = f"""
ОТЧЕТ ПО ВАЛИДАЦИОННЫМ ДАННЫМ
Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}
Модель: EfficientNet-B0

📊 СТАТИСТИКА:
• Всего обработано: {stats['total_processed']} изображений
• Изображений с низкой уверенностью (<60%): {stats['low_confidence']}

📈 РАСПРЕДЕЛЕНИЕ ПРЕДСКАЗАНИЙ:
"""

        for class_name, count in stats['by_class'].items():
            display_name = self.class_display_names[self.class_names.index(class_name)]
            report += f"  • {display_name}: {count} изображений\n"

        # Анализ правильности предсказаний
        if results:
            correct = sum(1 for r in results if r['correct'])
            accuracy = correct / len(results)

            report += f"""
🎯 ТОЧНОСТЬ НА ВАЛИДАЦИИ:
• Правильных предсказаний: {correct}/{len(results)}
• Точность: {accuracy:.2%}

📁 СТРУКТУРА ПАПОК:
• validation_predictions/ - корневая папка
  ├── no_defect/ - изображения, предсказанные как нормальные
  ├── crack/ - изображения, предсказанные как трещины
  ├── dent/ - изображения, предсказанные как вмятины
  ├── scratch/ - изображения, предсказанные как царапины
  ├── predictions_detailed.json - детальные результаты
  └── validation_report.txt - этот отчет
"""

        with open(output_path / "validation_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"📄 Отчет сохранен: {output_path}/validation_report.txt")


def create_sample_images_if_needed():
    """Создание тестовых изображений если валидационные данные отсутствуют"""
    val_path = Path("data/classification/val")

    if not val_path.exists():
        print("⚠️  Валидационные данные не найдены, создаем тестовые изображения...")

        # Создаем структуру папок
        for class_name in ['no_defect', 'crack', 'dent', 'scratch']:
            class_dir = val_path / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # Создаем по 2 тестовых изображения для каждого класса
            for i in range(2):
                # Создаем простое цветное изображение
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (224, 224), color=(100, 100, 100))
                draw = ImageDraw.Draw(img)

                # Добавляем метку класса
                draw.text((10, 10), f"Test {class_name} {i + 1}", fill=(255, 255, 255))

                # Сохраняем
                img.save(class_dir / f"test_{class_name}_{i + 1}.jpg")

        print(f"✅ Созданы тестовые изображения в {val_path}")


def main():
    """Основная функция"""
    print("=" * 70)
    print("🎨 ВИЗУАЛИЗАЦИЯ ПРЕДСКАЗАНИЙ НА ВАЛИДАЦИОННЫХ ДАННЫХ")
    print("=" * 70)

    # Создаем тестовые изображения если нужно
    create_sample_images_if_needed()

    # Инициализация визуализатора
    visualizer = PredictionVisualizer()

    # Обработка валидационных данных
    print("\n" + "=" * 70)
    print("1. ОБРАБОТКА ВАЛИДАЦИОННЫХ ДАННЫХ")
    print("=" * 70)

    results, stats = visualizer.process_validation_folder(
        input_folder="data/classification/val",
        output_folder="validation_predictions"
    )

    # Пример отдельных изображений
    print("\n" + "=" * 70)
    print("2. ПРИМЕРЫ ПРЕДСКАЗАНИЙ")
    print("=" * 70)

    # Ищем несколько примеров для демонстрации
    val_path = Path("data/classification/val")
    example_images = []

    if val_path.exists():
        for class_name in visualizer.class_names:
            class_dir = val_path / class_name
            if class_dir.exists():
                images = list(class_dir.glob('*.jpg'))
                if images:
                    example_images.append(images[0])

    if example_images:
        print(f"\n🔍 Примеры предсказаний для {len(example_images)} изображений:")

        for img_path in example_images[:3]:  # Показываем первые 3
            result, image = visualizer.predict_image(img_path)

            if result:
                print(f"\n📷 Изображение: {result['image_name']}")
                print(f"   Предсказанный класс: {result['predicted_class_display']}")
                print(f"   Уверенность: {result['confidence']:.2%}")

                # Определяем истинный класс из пути
                true_class = img_path.parent.name
                true_display = visualizer.class_display_names[visualizer.class_names.index(true_class)]
                print(f"   Истинный класс: {true_display}")

                if result['predicted_class'] == true_class:
                    print("   ✅ Предсказание верное!")
                else:
                    print("   ❌ Предсказание неверное")

    print("\n" + "=" * 70)
    print("✅ ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 70)

    print(f"\n📁 РЕЗУЛЬТАТЫ СОХРАНЕНЫ В ПАПКЕ: validation_predictions/")
    print("\n📊 СТАТИСТИКА:")
    print(f"   • Обработано изображений: {stats['total_processed']}")
    print(f"   • Низкая уверенность: {stats['low_confidence']}")

    print("\n🎯 КАК ИСПОЛЬЗОВАТЬ РЕЗУЛЬТАТЫ:")
    print("""
1. Откройте папку 'validation_predictions/'
2. В каждой подпапке (no_defect, crack, dent, scratch) находятся изображения
3. На каждом изображении есть подпись с предсказанным классом и уверенностью
4. Файл 'validation_report.txt' содержит статистику
5. Файл 'predictions_detailed.json' содержит детальные результаты
""")

    print("=" * 70)


if __name__ == "__main__":
    main()
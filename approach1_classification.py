# approach1_classification.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from config import config


class DefectDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = config.CLASS_NAMES
        self.class_to_idx = config.CLASS_TO_IDX

        self.images = []
        for class_name in self.classes:
            class_path = self.data_dir / class_name
            if class_path.exists():
                for img_path in class_path.glob('*.jpg'):
                    self.images.append((str(img_path), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def create_dataloaders():
    """Создание DataLoader'ов для train/val/test"""
    # Аугментации для тренировочных данных
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Только ресайз и нормализация для val/test
    val_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Датасеты
    train_dataset = DefectDataset(config.DATA_DIR / "train", train_transform)
    val_dataset = DefectDataset(config.DATA_DIR / "val", val_transform)
    test_dataset = DefectDataset(config.DATA_DIR / "test", val_transform)

    print(f"Train: {len(train_dataset)} images")
    print(f"Val: {len(val_dataset)} images")
    print(f"Test: {len(test_dataset)} images")

    # DataLoader'ы
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                             shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


def create_model(model_name='resnet50'):
    """Создание модели с transfer learning"""
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, config.NUM_CLASSES)
    elif model_name == 'mobilenet_v3':
        model = models.mobilenet_v3_large(pretrained=True)
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, config.NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(config.DEVICE)


def train_epoch(model, train_loader, criterion, optimizer, epoch):
    """Одна эпоха обучения"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'Loss': running_loss / (batch_idx + 1),
            'Acc': 100. * correct / total
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion):
    """Валидация"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = 100. * correct / total

    return val_loss, val_acc, all_preds, all_labels


def train_model():
    """Обучение модели"""
    print("=" * 60)
    print("ПОДХОД 1: КЛАССИФИКАЦИЯ С TRANSFER LEARNING")
    print("=" * 60)

    # Создаем даталоадеры
    train_loader, val_loader, test_loader = create_dataloaders()

    # Создаем модель
    model = create_model('efficientnet_b0')
    print(f"Model: EfficientNet-B0")
    print(f"Device: {config.DEVICE}")

    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     patience=3, factor=0.5)

    # Обучение
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("\nНачало обучения...")
    start_time = time.time()

    for epoch in range(config.NUM_EPOCHS):
        # Обучение
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)

        # Валидация
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion)

        # Сохраняем историю
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, config.MODEL_DIR / "best_classification_model.pth")

        # Обновляем learning rate
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    training_time = time.time() - start_time
    print(f"\nОбучение завершено за {training_time:.2f} секунд")

    # Загружаем лучшую модель
    checkpoint = torch.load(config.MODEL_DIR / "best_classification_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    # Тестирование на test set
    print("\nТестирование на test set...")
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # Сохраняем результаты
    results = {
        'model': 'EfficientNet-B0',
        'train_time': training_time,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'predictions': test_preds,
        'labels': test_labels
    }

    return model, results, history


def evaluate_model(model, test_loader):
    """Детальная оценка модели"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return all_preds, all_labels, all_probs


def create_classification_report(preds, labels, probs=None):
    """Создание отчета по классификации"""
    print("\n" + "=" * 60)
    print("ДЕТАЛЬНЫЙ ОТЧЕТ ПО КЛАССИФИКАЦИИ")
    print("=" * 60)

    # Classification report
    report = classification_report(labels, preds,
                                   target_names=config.CLASS_NAMES,
                                   output_dict=True)

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=config.CLASS_NAMES))

    # Confusion matrix
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "confusion_matrix_classification.png", dpi=150)
    plt.show()

    # Сохраняем метрики в DataFrame
    metrics_df = pd.DataFrame()
    for class_name in config.CLASS_NAMES:
        if class_name in report:
            metrics_df[class_name] = [
                report[class_name]['precision'],
                report[class_name]['recall'],
                report[class_name]['f1-score']
            ]

    metrics_df.index = ['Precision', 'Recall', 'F1-Score']

    # Общие метрики
    overall_metrics = {
        'accuracy': report['accuracy'],
        'macro_avg_precision': report['macro avg']['precision'],
        'macro_avg_recall': report['macro avg']['recall'],
        'macro_avg_f1': report['macro avg']['f1-score'],
        'weighted_avg_precision': report['weighted avg']['precision'],
        'weighted_avg_recall': report['weighted avg']['recall'],
        'weighted_avg_f1': report['weighted avg']['f1-score']
    }

    print("\nOverall Metrics:")
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics_df, overall_metrics


def plot_training_history(history):
    """Визуализация истории обучения"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['val_acc'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "training_history.png", dpi=150)
    plt.show()


def inference_speed_test(model, test_loader, n_tests=100):
    """Тест скорости инференса"""
    print("\nТестирование скорости инференса...")

    model.eval()

    # Подготовка одного батча для тестирования
    test_inputs, _ = next(iter(test_loader))
    test_inputs = test_inputs.to(config.DEVICE)

    # Прогрев GPU
    for _ in range(10):
        _ = model(test_inputs[:1])

    # Тестирование
    start_time = time.time()
    with torch.no_grad():
        for i in range(n_tests):
            _ = model(test_inputs[:1])

    inference_time = (time.time() - start_time) / n_tests
    fps = 1.0 / inference_time

    print(f"Время на один кадр: {inference_time * 1000:.2f} мс")
    print(f"Скорость: {fps:.2f} FPS")

    return inference_time, fps


def save_results(results_dict, filename="classification_results.json"):
    """Сохранение результатов"""
    import json

    # Конвертируем numpy в python типы
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return obj

    serializable_results = {}
    for key, value in results_dict.items():
        if key in ['predictions', 'labels']:
            serializable_results[key] = convert_to_serializable(value)
        else:
            serializable_results[key] = value

    with open(config.RESULTS_DIR / filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Результаты сохранены в {config.RESULTS_DIR / filename}")


def main():
    """Основная функция"""
    print("ЗАПУСК ПОДХОДА 1: КЛАССИФИКАЦИЯ")
    print("=" * 60)

    # 1. Обучение модели
    model, results, history = train_model()

    # 2. Детальная оценка
    print("\nДетальная оценка на тестовом наборе...")
    _, test_loader, _ = create_dataloaders()
    preds, labels, probs = evaluate_model(model, test_loader)
    results['predictions'] = preds
    results['labels'] = labels

    # 3. Отчет
    metrics_df, overall_metrics = create_classification_report(preds, labels, probs)
    results.update(overall_metrics)

    # 4. Визуализация истории обучения
    plot_training_history(history)

    # 5. Тест скорости
    inference_time, fps = inference_speed_test(model, test_loader)
    results['inference_time_ms'] = inference_time * 1000
    results['fps'] = fps

    # 6. Сохранение результатов
    save_results(results)

    # 7. Сохранение метрик
    metrics_df.to_csv(config.RESULTS_DIR / "classification_metrics.csv")
    print(f"\nМетрики сохранены в {config.RESULTS_DIR / 'classification_metrics.csv'}")

    print("\n" + "=" * 60)
    print("ПОДХОД 1 ЗАВЕРШЕН")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = main()
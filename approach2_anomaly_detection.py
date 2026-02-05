# approach2_simple_anomaly.py
"""
Упрощенный подход 2: Обнаружение аномалий с помощью Autoencoders
(Не требует anomalib)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import pandas as pd
import time
from tqdm import tqdm


class Autoencoder(nn.Module):
    """Простой автоэнкодер для обнаружения аномалий"""

    def __init__(self, input_dim=224):
        super(Autoencoder, self).__init__()

        # Энкодер
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 112x112
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 56x56
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 28x28
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 14x14
            nn.ReLU(),
        )

        # Декодер
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 224x224
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AnomalyDataset(Dataset):
    """Датасет для обнаружения аномалий"""

    def __init__(self, normal_dir, anomaly_dirs=None, transform=None, is_train=True):
        self.transform = transform
        self.is_train = is_train

        # Загружаем нормальные изображения
        normal_path = Path(normal_dir)
        self.normal_images = list(normal_path.glob('*.jpg'))

        # Для обучения используем только нормальные
        if is_train:
            self.images = self.normal_images
            self.labels = [0] * len(self.images)  # 0 = нормальные
        else:
            # Для тестирования добавляем аномальные
            self.anomaly_images = []
            if anomaly_dirs:
                for dir_path in anomaly_dirs:
                    anomaly_path = Path(dir_path)
                    if anomaly_path.exists():
                        self.anomaly_images.extend(list(anomaly_path.glob('*.jpg')))

            self.images = self.normal_images + self.anomaly_images
            self.labels = [0] * len(self.normal_images) + [1] * len(self.anomaly_images)

        print(f"{'Train' if is_train else 'Test'}: {len(self.normal_images)} нормальных, "
              f"{len(self.anomaly_images) if not is_train else 0} аномальных")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label, str(img_path)


def train_autoencoder():
    """Обучение автоэнкодера на нормальных изображениях"""
    print("=" * 60)
    print("ПОДХОД 2: ОБНАРУЖЕНИЕ АНОМАЛИЙ (Autoencoder)")
    print("=" * 60)

    # Параметры
    BATCH_SIZE = 32
    IMG_SIZE = 224
    EPOCHS = 50
    LEARNING_RATE = 0.001

    # Трансформы
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # Данные для обучения (только нормальные)
    train_dataset = AnomalyDataset(
        normal_dir="data/classification/train/no_defect",
        transform=transform,
        is_train=True
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Данные для тестирования (нормальные + аномальные)
    test_dataset = AnomalyDataset(
        normal_dir="data/classification/test/no_defect",
        anomaly_dirs=[
            "data/classification/test/crack",
            "data/classification/test/dent",
            "data/classification/test/scratch"
        ],
        transform=transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ]),
        is_train=False
    )

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Модель и оптимизатор
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Обучение
    print("\n🎯 Обучение автоэнкодера на нормальных изображениях...")
    train_losses = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}')
        for images, _, _ in pbar:
            images = images.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'Loss': running_loss / len(pbar)})

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.6f}")

    print(f"\n✅ Обучение завершено. Final loss: {train_losses[-1]:.6f}")

    # Сохраняем модель
    torch.save(model.state_dict(), "models/autoencoder_model.pth")

    return model, device, test_loader, train_losses


def evaluate_anomaly_detection(model, device, test_loader):
    """Оценка модели обнаружения аномалий"""
    print("\n🔍 Оценка обнаружения аномалий...")

    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc="Оценка"):
            images = images.to(device)

            # Реконструкция
            reconstructed = model(images)

            # Вычисляем MSE между оригиналом и реконструкцией
            mse = torch.mean((images - reconstructed) ** 2, dim=[1, 2, 3])

            all_scores.extend(mse.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Нормализуем scores
    all_scores = (all_scores - all_scores.min()) / (all_scores.max() - all_scores.min() + 1e-8)

    return all_scores, all_labels


def calculate_metrics(scores, labels):
    """Расчет метрик для обнаружения аномалий"""
    print("\n📊 Расчет метрик...")

    # ROC-AUC
    roc_auc = roc_auc_score(labels, scores)

    # Precision-Recall AUC
    pr_auc = average_precision_score(labels, scores)

    # Находим оптимальный порог
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

    # Предсказания с оптимальным порогом
    predictions = (scores > optimal_threshold).astype(int)

    # Basic metrics
    accuracy = np.mean(predictions == labels)
    precision = np.sum((predictions == 1) & (labels == 1)) / (np.sum(predictions == 1) + 1e-8)
    recall = np.sum((predictions == 1) & (labels == 1)) / (np.sum(labels == 1) + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Optimal threshold: {optimal_threshold:.4f}")

    return {
               'roc_auc': roc_auc,
               'pr_auc': pr_auc,
               'accuracy': accuracy,
               'precision': precision,
               'recall': recall,
               'f1_score': f1,
               'optimal_threshold': optimal_threshold,
               'optimal_f1': f1_scores[optimal_idx]
           }, scores, labels, predictions


def visualize_results(scores, labels, metrics, train_losses):
    """Визуализация результатов"""
    # Создаем папку для результатов
    Path("results/anomaly").mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Распределение anomaly scores
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    axes[0, 0].hist(normal_scores, bins=30, alpha=0.7, label='Normal', color='green')
    axes[0, 0].hist(anomaly_scores, bins=30, alpha=0.7, label='Anomaly', color='red')
    axes[0, 0].axvline(metrics['optimal_threshold'], color='black', linestyle='--',
                       label=f"Threshold: {metrics['optimal_threshold']:.3f}")
    axes[0, 0].set_xlabel('Anomaly Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Anomaly Scores')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Loss during training
    axes[0, 1].plot(train_losses)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss')
    axes[0, 1].set_title('Training Loss (Autoencoder)')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)

    axes[0, 2].plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})')
    axes[0, 2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 2].set_xlim([0.0, 1.0])
    axes[0, 2].set_ylim([0.0, 1.05])
    axes[0, 2].set_xlabel('False Positive Rate')
    axes[0, 2].set_ylabel('True Positive Rate')
    axes[0, 2].set_title('ROC Curve')
    axes[0, 2].legend(loc="lower right")
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Precision-Recall Curve
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(labels, scores)

    axes[1, 0].plot(recall, precision, color='blue', lw=2,
                    label=f'PR curve (AUC = {metrics["pr_auc"]:.3f})')
    axes[1, 0].set_xlim([0.0, 1.0])
    axes[1, 0].set_ylim([0.0, 1.05])
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curve')
    axes[1, 0].legend(loc="lower left")
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Confusion Matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    predictions = (scores > metrics['optimal_threshold']).astype(int)
    cm = confusion_matrix(labels, predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
    disp.plot(ax=axes[1, 1], cmap='Blues')
    axes[1, 1].set_title(f'Confusion Matrix (threshold={metrics["optimal_threshold"]:.3f})')

    # 6. Примеры реконструкций
    axes[1, 2].axis('off')
    axes[1, 2].text(0.5, 0.5, 'Примеры реконструкций\nбудут в отдельном файле',
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[1, 2].transAxes, fontsize=12)

    plt.tight_layout()
    plt.savefig("results/anomaly/anomaly_detection_results.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Отдельно: примеры реконструкций
    plot_reconstruction_examples()


def plot_reconstruction_examples():
    """Визуализация примеров реконструкций"""
    # Загружаем несколько изображений
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Загружаем модель
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load("models/autoencoder_model.pth", map_location=device))
    model.eval()

    # Выбираем несколько изображений
    test_images = []
    test_labels = []

    # Нормальное
    normal_dir = Path("data/classification/test/no_defect")
    if normal_dir.exists():
        normal_imgs = list(normal_dir.glob('*.jpg'))[:2]
        test_images.extend(normal_imgs)
        test_labels.extend(['normal'] * 2)

    # Аномальные
    for defect_dir in ["crack", "dent", "scratch"]:
        defect_path = Path("data/classification/test") / defect_dir
        if defect_path.exists():
            defect_imgs = list(defect_path.glob('*.jpg'))[:1]
            test_images.extend(defect_imgs)
            test_labels.extend([defect_dir] * 1)

    # Создаем визуализацию
    fig, axes = plt.subplots(len(test_images), 3, figsize=(10, len(test_images) * 3))

    if len(test_images) == 1:
        axes = axes.reshape(1, -1)

    for idx, (img_path, label) in enumerate(zip(test_images, test_labels)):
        # Загружаем изображение
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Реконструкция
        with torch.no_grad():
            reconstructed = model(img_tensor)

        # Конвертируем обратно для отображения
        original_img = img_tensor[0].cpu().permute(1, 2, 0).numpy()
        recon_img = reconstructed[0].cpu().permute(1, 2, 0).numpy()

        # Разница
        diff_img = np.abs(original_img - recon_img)
        diff_img = diff_img / diff_img.max()  # Нормализуем

        # Показываем
        axes[idx, 0].imshow(original_img)
        axes[idx, 0].set_title(f"Original: {label}")
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(recon_img)
        axes[idx, 1].set_title("Reconstructed")
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(diff_img, cmap='hot')
        axes[idx, 2].set_title("Difference (anomaly)")
        axes[idx, 2].axis('off')

    plt.suptitle('Примеры реконструкций автоэнкодера', fontsize=14)
    plt.tight_layout()
    plt.savefig("results/anomaly/reconstruction_examples.png", dpi=150, bbox_inches='tight')
    plt.show()


def save_results(metrics, training_time):
    """Сохранение результатов"""
    results = {
        'approach': 'Anomaly Detection (Autoencoder)',
        'model': 'Simple Autoencoder',
        'training_time': training_time,
        'metrics': metrics
    }

    import json
    with open("results/anomaly/anomaly_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Также в CSV для сравнения
    df = pd.DataFrame([metrics])
    df.to_csv("results/anomaly/anomaly_metrics.csv", index=False)

    print(f"\n📁 Результаты сохранены:")
    print("   - models/autoencoder_model.pth")
    print("   - results/anomaly/anomaly_results.json")
    print("   - results/anomaly/anomaly_metrics.csv")
    print("   - results/anomaly/anomaly_detection_results.png")
    print("   - results/anomaly/reconstruction_examples.png")


def main():
    """Основная функция"""
    print("=" * 60)
    print("ЗАПУСК ПОДХОДА 2: ОБНАРУЖЕНИЕ АНОМАЛИЙ")
    print("=" * 60)

    start_time = time.time()

    # 1. Обучение
    model, device, test_loader, train_losses = train_autoencoder()

    training_time = time.time() - start_time
    print(f"\n⏱️  Время обучения: {training_time:.2f} секунд")

    # 2. Оценка
    scores, labels = evaluate_anomaly_detection(model, device, test_loader)

    # 3. Расчет метрик
    metrics, scores, labels, predictions = calculate_metrics(scores, labels)

    # 4. Визуализация
    visualize_results(scores, labels, metrics, train_losses)

    # 5. Сохранение результатов
    save_results(metrics, training_time)

    print("\n" + "=" * 60)
    print("ПОДХОД 2 ЗАВЕРШЕН")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    metrics = main()
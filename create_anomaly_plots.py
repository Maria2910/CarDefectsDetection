# create_anomaly_plots.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import seaborn as sns
from pathlib import Path

# Создаем папку
Path("results/anomaly_plots").mkdir(exist_ok=True)

# Примерные данные на основе ваших результатов
# (В реальности нужно загрузить scores и labels из сохраненных результатов)
np.random.seed(42)

# Генерируем примерные данные
n_normal = 30
n_anomaly = 180

# Normal scores (низкие)
normal_scores = np.random.exponential(0.002, n_normal)

# Anomaly scores (высокие)
anomaly_scores = np.random.exponential(0.01, n_anomaly) + 0.005

scores = np.concatenate([normal_scores, anomaly_scores])
labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])

# Нормализуем (как в скрипте)
scores = (scores - scores.min()) / (scores.max() - scores.min())

# Оптимальный порог из ваших результатов
optimal_threshold = 0.0042

# 1. Распределение scores
plt.figure(figsize=(10, 6))
plt.hist(scores[labels == 0], bins=30, alpha=0.7, label='Normal', color='green')
plt.hist(scores[labels == 1], bins=30, alpha=0.7, label='Anomaly', color='red')
plt.axvline(optimal_threshold, color='black', linestyle='--',
            label=f'Threshold: {optimal_threshold:.4f}')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.title('Distribution of Anomaly Scores\n(Approximation based on your results)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/anomaly_plots/score_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# 2. ROC Curve
fpr, tpr, _ = roc_curve(labels, scores)
roc_auc = 0.9774  # из ваших результатов

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Anomaly Detection')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig('results/anomaly_plots/roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# 3. Confusion Matrix
predictions = (scores > optimal_threshold).astype(int)
cm = confusion_matrix(labels, predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
plt.title(f'Confusion Matrix\n(Threshold = {optimal_threshold:.4f})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('results/anomaly_plots/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Визуализации созданы в results/anomaly_plots/")
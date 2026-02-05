# explore_data.py
import os
import glob
from pathlib import Path

# Путь к распакованным данным
data_path = "NEU-DET"  # Обычно датасет называется так

if not os.path.exists(data_path):
    # Если папка называется иначе, поищем её
    possible_names = ["NEU-DET", "neu-det", "NEU_DET", "neu_surface_defect_database"]
    for name in possible_names:
        if os.path.exists(name):
            data_path = name
            break
    else:
        # Посмотрим какие папки есть в текущей директории
        print("Содержимое текущей директории:")
        for item in os.listdir('.'):
            if os.path.isdir(item):
                print(f"  - {item}/")
        raise FileNotFoundError(f"Не найдена папка с данными. Проверьте название распакованной папки.")

print(f"Исследуем папку: {data_path}")
print("\nСодержимое папки:")
for root, dirs, files in os.walk(data_path):
    level = root.replace(data_path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # Покажем только первые 5 файлов
        print(f'{subindent}{file}')
    if len(files) > 5:
        print(f'{subindent}... и еще {len(files) - 5} файлов')
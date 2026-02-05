# quick_test.py
import importlib

required_packages = [
    'torch',
    'torchvision',
    'sklearn',
    'pandas',
    'matplotlib',
    'seaborn',
    'cv2',
    'tqdm',
    'anomalib',
    'ultralytics'
]

print("Проверка установленных пакетов:")
print("-" * 40)

for package in required_packages:
    try:
        if package == 'sklearn':
            importlib.import_module('sklearn')
            print(f"✅ scikit-learn (sklearn)")
        elif package == 'cv2':
            importlib.import_module('cv2')
            print(f"✅ opencv-python (cv2)")
        else:
            importlib.import_module(package)
            print(f"✅ {package}")
    except ImportError:
        print(f"❌ {package} - не установлен")

print("-" * 40)
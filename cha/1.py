import os
import subprocess

# Путь к вашему проекту YOLOv8
yolo_project_dir = r"C:\Users\User\AppData\Local\Programs\Python\Python311\Lib\site-packages"

# Путь к вашему YAML файлу с данными
data_yaml_path = r"C:\cha\dataset\data.yaml"

# Гиперпараметры обучения
epochs = 100  # Количество эпох
batch_size = 16  # Размер батча
img_size = 640  # Размер изображения

# Путь к модели (в данном случае YOLOv8n)
model = "yolov8n.pt"

# Команда для тренировки
train_command = [
    "yolo", "train", 
    f"model={model}",
    f"data={data_yaml_path}",
    f"epochs={epochs}",
    f"batch={batch_size}",
    f"imgsz={img_size}"
]

# Запуск команды в терминале
def train_yolov8():
    try:
        # Изменение директории на директорию проекта YOLOv8
        os.chdir(yolo_project_dir)
        
        # Запуск процесса тренировки
        print("Starting YOLOv8 training...")
        subprocess.run(train_command, check=True)
        
        print("Training complete.")
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    train_yolov8()

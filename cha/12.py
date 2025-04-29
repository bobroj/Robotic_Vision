import cv2
from ultralytics import YOLO
import numpy as np

# Загружаем модель C:\Users\User\AppData\Local\Programs\Python\Python311\Lib\site-packages\runs\detect\train
model = YOLO('C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\runs\\detect\\train3\\weights\\best.pt')

# Загружаем тестовое изображение
img = cv2.imread("msq.jpg")
img_height, img_width = img.shape[:2]

# Применяем модель для предсказания
results = model(img)

# Получаем аннотированное изображение
annotated_img = results[0].plot()

# Наносим координатную сетку 8x8 (шахматная доска)
cell_w = 100
cell_h = 100

w = img_width//100
h = img_height//100

for i in range(1, w+1):
    cv2.line(annotated_img, (i * cell_w, 0), (i * cell_w, img_height), (0, 255, 0), 1)
    cv2.line(annotated_img, (0, i * cell_h), (img_width, i * cell_h), (0, 255, 0), 1)

# Пишем координаты и метки объектов
names = results[0].names
boxes = results[0].boxes

for box in boxes:
    cls_id = int(box.cls.item())
    label = names[cls_id]
    x, y, w, h = box.xywh.cpu().numpy()[0]
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)
    
    # Рисуем метку на изображении
    #cv2.putText(annotated_img, f'{label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    #cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    print(f"{label} на координатах x: {x:.2f}, y: {y:.2f}, ширина: {w:.2f}, высота: {h:.2f}")

# Показываем изображение с предсказаниями и сеткой
cv2.imshow('Predicted Image with Grid', annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()








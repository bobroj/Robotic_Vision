import cv2
from ultralytics import YOLO

# Загружаем модель
model = YOLO(r"C:\Users\User\AppData\Local\Programs\Python\Python311\Lib\site-packages\runs\detect\train\weights\best.pt")

# Открываем видео (можно указать путь к файлу или 0 для вебкамеры)
cap = cv2.VideoCapture(0)  # или cap = cv2.VideoCapture(0) для камеры

while True:
    ret, frame = cap.read()
    if not ret:
        break  # если видео закончилось

    # Уменьшаем размер (по желанию)
    frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)

    # Предсказание YOLO
    results = model(frame, verbose=False)

    # Отображаем результаты на кадре
    annotated_frame = results[0].plot()  # аннотированный кадр

    # Показываем кадр
    cv2.imshow("YOLO Detection", annotated_frame)

    # Выход по клавише "q"
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


from ultralytics import YOLO

# Загружаем модель
model = YOLO(r"C:\Users\User\AppData\Local\Programs\Python\Python311\Lib\site-packages\runs\detect\train3\weights\best.pt")

# Меняем имена классов
model.model.names = {
    0: 'b',
    1: 'k',
    2: 'n',
    3: 'p',
    4: 'q',
    5: 'r',
    6: 'B',
    7: 'K',
    8: 'N',
    9: 'P',
    10: 'Q',
    11: 'R'
}

# Сохраняем правильно — С ЭТИМИ именами
model.save(r"C:\Users\User\AppData\Local\Programs\Python\Python311\Lib\site-packages\runs\detect\train3\weights\best.pt")
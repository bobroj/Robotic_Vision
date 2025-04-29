import cv2
from ultralytics import YOLO
import os

import numpy as np
import chess
import chess.engine

from detect_points import get_points

highlight_move_coords = None

engine = chess.engine.SimpleEngine.popen_uci(r"C:\Users\KSAS2\Downloads\stockfish-windows-x86-64-avx2 (1)\stockfish\stockfish-windows-x86-64-avx2")

dir_path = os.path.dirname(os.path.realpath(file))+"/numpy_saved" # path of current directory
boxes = np.zeros((8,8,4),dtype=int)

# Загружаем модель
model = YOLO(r"C:\train2\train3\weights\best.pt")



cap = cv2.VideoCapture(2)  # 0 — это номер камеры (можно попробовать 1, если не работает)

while True:
    
    ret, frame = cap.read()  # Чтение кадра
    
    if not ret:
        print("Error of camera connection")
        break
    cv2.imshow("dsf", frame)
    
    points = []
    

    if cv2.waitKey(3) & 0xFF == ord('q'):  # Выход по клавише 'q'
        break



###################################################################################
## calibrate points for chess corners
###################################################################################
while True:
        print("do you want to calibrate new Points for corners [y/n]:",end=" ")
        ans = str(input())
        if ans == "y" or ans == "Y":
            ret , img = cv2.VideoCapture(2).read()
            #img =   cv2.resize(img,(800,800))
            #img = get_warp_img(img,dir_path,img_resize)
            points = []
            for i in range(9):
                pt = get_points(img,9)
                points.append(pt)
            np.savez(dir_path+"/chess_board_points.npz",points=points)
            break
        elif ans == "n" or ans == "N":
            # do some work
            points = np.load(dir_path+'/chess_board_points.npz')['points']
            print("points Load successfully")
            break
        else:
            print("something wrong input")

###################################################################################
## Define Boxes
###################################################################################
for i in range(8):
    for j in range(8):
        boxes[i][j][0] = points[i][j][0]
        boxes[i][j][1] = points[i][j][1]
        boxes[i][j][2] = points[i+1][j+1][0]
        boxes[i][j][3] = points[i+1][j+1][1]

np.savez(dir_path+"/chess_board_Box.npz",boxes=boxes)


###################################################################################
## View Boxes
###################################################################################
while True:
    print("Do you want to see Boxex on Chess board [y/n]:",end=" ")
    ans = str(input())
    if ans == 'y' or ans == "Y":
        # show boxes
        ret , img = cv2.VideoCapture(2).read()
        #img =   cv2.resize(img,(800,800))
        #img = get_warp_img(img,dir_path,img_resize)
        img_box = img.copy()
        for i in range(8):
            for j in range(8):
                box1 = boxes[i,j]
                cv2.rectangle(img_box, (int(box1[0]), int(box1[1])), (int(box1[2]), int(box1[3])), (255,0,0), 2)
                cv2.putText(img_box,"({},{})".format(i,j),(int(box1[2])-70, int(box1[3])-50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                cv2.imshow("img",img_box)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
    elif ans == 'N' or ans == "n":
        print("ok, got it you don't want ot see boxes")
        break
    else:
        print("Enter valid input")


cap.release()  # Освобождаем камеру
cv2.destroyAllWindows()  # Закрываем окно


def find_square_containing_point(x, y, boxes):
    """
    Возвращает (i, j), если точка (x, y) лежит в клетке boxes[i][j]
    """
    for i in range(8):
        for j in range(8):
            x1, y1, x2, y2 = boxes[i][j]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return (i, j)
    return None  # если не нашли

board_final = [['.' for _ in range(8)] for _ in range(8)]
def cells_coordinates(names, yolo_boxes, board_final):
    for box in yolo_boxes:
        cls_id = int(box.cls.item())
        label = names[cls_id]
        x, y, w, h = box.xywh.cpu().numpy()[0]
        y = int(y + h/4)

        """
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        """
   

        print(f"{label} With coordinates x: {x:.2f}, y: {y:.2f}, width: {w:.2f}, height: {h:.2f}")
        
        cell = find_square_containing_point(x,y,boxes)
        if cell:
            i,j = cell
            print({i}, {j})
            board_final[i][j] = label
        else:
            print("out of cell!!!")


def board_to_fen(board):
    fen_rows = []
    for row in board:
        empty_count = 0
        fen_row = ""
        for cell in row:
            if cell == ".":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    return "/".join(fen_rows) + " w - - 0 1"  




# Открываем видео (можно указать путь к файлу или 0 для вебкамеры)
cap = cv2.VideoCapture(2)  # или cap = cv2.VideoCapture(0) для камеры


import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk



def update_frames():
    global highlight_move_coords
    ret, frame = cap.read()
    if not ret:
        root.after(30, update_frames)
        return


    # YOLO-аннотация справа
    results = model(frame, verbose=False)
    frame_right = results[0].plot()

    height, width = frame_right.shape[:2]
    frame_right = cv2.resize(frame_right, (width // 1, height // 1))

    #frame_right = cv2.resize(frame_right, (300, 300))
    img_right = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)))
    label_yolo.config(image=img_right)
    label_yolo.image = img_right


    # Видео слева
    frame_left = frame
    if highlight_move_coords is not None:
        for (x1, y1, x2, y2) in highlight_move_coords:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            radius = 10
            # Преобразуй координаты в масштаб left_frame
            #cx = cx // 2
            #cy = cy // 2
            cv2.circle(frame_left, (cx, cy), radius, (0, 255, 0), -1)

    height, width = frame_left.shape[:2]
    frame_left = cv2.resize(frame, (0, 0), fx=1/1, fy=1/1)

    #frame_left = cv2.resize(frame, (width // 2, height // 2))
    #frame_left = cv2.resize(frame, (400, 300))
    img_left = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)))
    label_main.config(image=img_left)
    label_main.image = img_left

    

    root.after(800, update_frames)



def square_to_index(from_square, to_square):  # для повернутой вправо доски
    col1 = ord(from_square[0]) - ord('a')     # 'e' → 4
    row1 = 8 - int(from_square[1])            # '2' → 6

    col2 = ord(to_square[0]) - ord('a')       
    row2 = 8 - int(to_square[1])

    # Поворот вправо на 90°:
    new_row1, new_col1 = col1, 7 - row1
    new_row2, new_col2 = col2, 7 - row2

    return new_row1, new_col1, new_row2, new_col2



def draw_transparent_rect(img, pt1, pt2, color=(0,255,0), alpha=0.5):
    overlay = img.copy()

    # Находим центр между pt1 и pt2
    center_x = (pt1[0] + pt2[0]) // 2
    center_y = (pt1[1] + pt2[1]) // 2

    # Рисуем круг на overlay
    radius = min(abs(pt2[0] - pt1[0]), abs(pt2[1] - pt1[1])) // 4
    cv2.circle(overlay, (center_x, center_y), radius, color, -1)

    # Добавляем прозрачность
    frame = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Показываем результат
    cv2.imshow("ds", frame)



def get_best_move():
    ret, frame = cap.read()
    snapshot = frame.copy()
    result = model(snapshot, verbose=False)

    print("processing..")

    board_final = [['.' for _ in range(8)] for _ in range(8)]
    names = result[0].names
    yolo_boxes = result[0].boxes
    
    cells_coordinates(names, yolo_boxes, board_final)
    rotated_board = [list(row) for row in zip(*board_final)][::-1]
    print(rotated_board)

    fen_string = board_to_fen(rotated_board)
    print(fen_string)

    board_st = chess.Board()
    board_st.set_fen(fen_string)
    best_move = engine.play(board_st, chess.engine.Limit(time=2))

    
    best_move_label.config(text=f"Best move: {best_move.move.uci()}")
    from_square = best_move.move.uci()[:2]
    to_square = best_move.move.uci()[2:]
    print(from_square)
    print(to_square)

    row1, col1, row2, col2 = square_to_index(from_square, to_square)
    #x10, y10, x20, y20 = boxes[row1][col1]
    #x11, y11, x21, y21 = boxes[row2][col2]
    global highlight_move_coords
    highlight_move_coords = [boxes[row1][col1], boxes[row2][col2]]
    
    #draw_transparent_rect(frame, (x11, y11), (x21, y21), color=(0, 255, 0), alpha=0.6)



root = tk.Tk()
root.title("Chess Vision")
root.geometry("800x600")
root.configure(bg="black")

label_main = Label(root, bg="black", highlightthickness=0)
label_main.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")

label_yolo = Label(root, bg="black", highlightthickness=0)
label_yolo.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

btn = Button(root, text="Get Best Move", command=get_best_move,
             bg="red", fg="white", height=2, width=20)
btn.grid(row=1, column=1, padx=10, pady=10)

best_move_label = Label(root, text=f"Best move: ", font=("Arial", 16),
                        bg="black", fg="white")
best_move_label.grid(row=2, column=1, columnspan=2, pady=10)

# Настройка сетки
root.grid_columnconfigure(0, weight=2)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=0)

update_frames()
root.mainloop()

# Очистка
cap.release()
engine.quit()
cv2.destroyAllWindows()
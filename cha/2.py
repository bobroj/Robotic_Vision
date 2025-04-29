import torch
print(torch.cuda.is_available())  # Должно вернуть True, если CUDA доступен
print(torch.__version__)  # Проверим, что версия корректная



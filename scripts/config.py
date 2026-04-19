"""
Конфигурационный файл для проекта.
Содержит все параметры для воспроизводимого обучения.
"""

import torch

class Config:
    # ========== ПУТИ К ДАННЫМ ==========
    DATA_PATH = "data/"
    IMAGES_PATH = "data/images/"
    DISH_CSV = "data/dish.csv"
    INGREDIENTS_CSV = "data/ingredients.csv"
    
    # ========== ПАРАМЕТРЫ МОДЕЛИ ==========
    BACKBONE = 'resnet18'        # resnet18, resnet34, resnet50
    PRETRAINED = True            # использовать предобученные веса
    EMBEDDING_DIM = 128          # размерность эмбеддингов ингредиентов
    MAX_INGREDIENTS = 30         # максимальное количество ингредиентов (padding)
    DROPOUT_RATE = 0.3           # dropout для регуляризации
    
    # ========== ПАРАМЕТРЫ ИЗОБРАЖЕНИЙ ==========
    IMG_SIZE = 224               # размер после ресайза (224×224)
    IMG_MEAN = [0.485, 0.456, 0.406]   # ImageNet mean
    IMG_STD = [0.229, 0.224, 0.225]    # ImageNet std
    
    # ========== ГИПЕРПАРАМЕТРЫ ОБУЧЕНИЯ ==========
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    WEIGHT_DECAY = 1e-4          # L2 регуляризация
    
    # ========== РАННЯЯ ОСТАНОВКА ==========
    EARLY_STOPPING_PATIENCE = 10  # сколько эпох ждать улучшения
    EARLY_STOPPING_MIN_DELTA = 0.001  # минимальное улучшение
    
    # ========== ПУТИ ДЛЯ СОХРАНЕНИЯ ==========
    MODEL_SAVE_PATH = "models/best_model.pth"
    LOGS_PATH = "logs/"
    RESULTS_PATH = "results/"
    
    # ========== ВОСПРОИЗВОДИМОСТЬ ==========
    SEED = 42
    
    # ========== ЛОГИРОВАНИЕ ==========
    LOG_INTERVAL = 10            # печатать лог каждые N батчей
    SAVE_CHECKPOINT = True       # сохранять чекпоинты
    
    # ========== ДОПОЛНИТЕЛЬНЫЕ НАСТРОЙКИ ==========
    NUM_WORKERS = 4              # количество потоков для загрузки данных
    USE_GPU = True               # использовать GPU если доступен
    
    @classmethod
    def get_device(cls):
        """Возвращает устройство (cuda/cpu)"""
        if cls.USE_GPU and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    
    @classmethod
    def to_dict(cls):
        """Преобразует конфиг в словарь для логирования"""
        return {key: value for key, value in cls.__dict__.items() 
                if not key.startswith('__') and not callable(value)}

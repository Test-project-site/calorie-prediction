"""
Вспомогательные функции для обучения:
- установка seed для воспроизводимости
- логирование метрик
- ранняя остановка
- сохранение чекпоинтов
"""

import os
import random
import numpy as np
import torch
import json
from datetime import datetime

def set_seed(seed):
    """
    Устанавливает seed для воспроизводимости результатов.
    
    Args:
        seed: целое число для инициализации генераторов случайных чисел
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Для детерминированных операций на GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Seed {seed} установлен для всех генераторов")


class EarlyStopping:
    """Ранняя остановка обучения при отсутствии улучшений"""
    
    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        """
        Args:
            patience: сколько эпох ждать улучшения
            min_delta: минимальное изменение для считания улучшением
            verbose: печатать ли сообщения
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss, epoch, model, save_path):
        """
        Проверяет, нужно ли остановить обучение.
        
        Returns:
            should_stop: bool
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self._save_checkpoint(model, save_path, val_loss, epoch)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self._save_checkpoint(model, save_path, val_loss, epoch)
            self.counter = 0
        
        return self.early_stop
    
    def _save_checkpoint(self, model, save_path, val_loss, epoch):
        """Сохраняет лучшую модель"""
        if self.verbose:
            print(f"Сохранение лучшей модели (epoch {epoch}, val_loss: {val_loss:.4f})")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
        }, save_path)


class Logger:
    """Логирование метрик обучения"""
    
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'learning_rates': [],
            'epoch_times': []
        }
        os.makedirs(log_dir, exist_ok=True)
    
    def add_metrics(self, epoch, train_loss, val_loss, train_mae, val_mae, lr, epoch_time):
        """Добавляет метрики за эпоху"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_mae'].append(train_mae)
        self.history['val_mae'].append(val_mae)
        self.history['learning_rates'].append(lr)
        self.history['epoch_times'].append(epoch_time)
    
    def save(self, filename='training_history.json'):
        """Сохраняет историю обучения в JSON"""
        save_path = os.path.join(self.log_dir, filename)
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"История обучения сохранена в {save_path}")
    
    def get_best_epoch(self):
        """Возвращает номер эпохи с наименьшей val_loss"""
        return np.argmin(self.history['val_loss']) + 1
    
    def get_best_val_loss(self):
        """Возвращает наименьшую val_loss"""
        return min(self.history['val_loss'])
    
    def get_best_val_mae(self):
        """Возвращает наименьшую val_mae"""
        return min(self.history['val_mae'])


def compute_metrics(predictions, targets):
    """
    Вычисляет метрики для регрессии.
    
    Args:
        predictions: numpy array предсказаний
        targets: numpy array истинных значений
    
    Returns:
        dict: словарь с метриками
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # MAE - Mean Absolute Error
    mae = np.mean(np.abs(predictions - targets))
    
    # MSE - Mean Squared Error
    mse = np.mean((predictions - targets) ** 2)
    
    # RMSE - Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # MAPE - Mean Absolute Percentage Error
    # Защита от деления на ноль
    mask = targets != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((predictions[mask] - targets[mask]) / targets[mask])) * 100
    else:
        mape = float('inf')
    
    # R² - Coefficient of Determination
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }


def print_metrics(metrics, prefix=""):
    """Печатает метрики в читаемом формате"""
    prefix_str = f"{prefix} " if prefix else ""
    print(f"{prefix_str}MAE: {metrics['mae']:.4f} ккал")
    print(f"{prefix_str}RMSE: {metrics['rmse']:.4f} ккал")
    print(f"{prefix_str}R²: {metrics['r2']:.4f}")
    print(f"{prefix_str}MAPE: {metrics['mape']:.2f}%")


def save_config(config, save_path='config_saved.json'):
    """Сохраняет конфигурацию в JSON файл"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2, default=str)
    print(f"Конфигурация сохранена в {save_path}")

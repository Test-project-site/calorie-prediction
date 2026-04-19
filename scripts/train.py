"""
Основной файл обучения.
Содержит функцию train(), которая запускает полный цикл обучения.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from .config import Config
from .dataset import FoodDataset
from .model import CaloriePredictor
from .utils import (
    set_seed, EarlyStopping, Logger, 
    compute_metrics, print_metrics, save_config
)


def get_transforms(config, split='train'):
    """
    Возвращает трансформации для изображений.
    
    Args:
        config: объект конфигурации
        split: 'train' или 'test'
    
    Returns:
        compose: композиция трансформаций
    """
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD)
        ])
    
    return transform


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Обучает модель одну эпоху.
    
    Returns:
        loss: средняя loss за эпоху
        predictions: список предсказаний
        targets: список истинных значений
    """
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for batch in pbar:
        images = batch['image'].to(device)
        ingredients = batch['ingredients'].to(device)
        calories = batch['calories'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images, ingredients)
        loss = criterion(outputs, calories)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions.extend(outputs.cpu().detach().numpy())
        targets.extend(calories.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, predictions, targets


def validate_epoch(model, dataloader, criterion, device):
    """
    Валидирует модель на одну эпоху.
    
    Returns:
        loss: средняя loss за эпоху
        predictions: список предсказаний
        targets: список истинных значений
    """
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation', leave=False)
        for batch in pbar:
            images = batch['image'].to(device)
            ingredients = batch['ingredients'].to(device)
            calories = batch['calories'].to(device)
            
            outputs = model(images, ingredients)
            loss = criterion(outputs, calories)
            
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            targets.extend(calories.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, predictions, targets


def train(config, resume_from=None):
    """
    Основная функция обучения.
    
    Args:
        config: объект конфигурации
        resume_from: путь к чекпоинту для возобновления обучения
    
    Returns:
        model: обученная модель
        logger: объект с историей обучения
    """
    print("=" * 60)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 60)
    
    # ========== ВОСПРОИЗВОДИМОСТЬ ==========
    set_seed(config.SEED)
    
    # ========== УСТРОЙСТВО ==========
    device = config.get_device()
    print(f"Используемое устройство: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # ========== ТРАНСФОРМАЦИИ ==========
    train_transform = get_transforms(config, 'train')
    val_transform = get_transforms(config, 'test')
    
    # ========== ЗАГРУЗЧИКИ ДАННЫХ ==========
    print("\nЗагрузка данных...")
    train_dataset = FoodDataset(
        csv_path=config.DISH_CSV,
        images_path=config.IMAGES_PATH,
        split='train',
        transform=train_transform,
        config=config
    )
    
    val_dataset = FoodDataset(
        csv_path=config.DISH_CSV,
        images_path=config.IMAGES_PATH,
        split='test',
        transform=val_transform,
        config=config
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # ========== МОДЕЛЬ ==========
    print("\nСоздание модели...")
    model = CaloriePredictor(
        num_ingredients=train_dataset.num_ingredients,
        embedding_dim=config.EMBEDDING_DIM,
        backbone=config.BACKBONE,
        pretrained=config.PRETRAINED,
        dropout_rate=config.DROPOUT_RATE
    )
    model.to(device)
    
    # Подсчёт параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Всего параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")
    
    # ========== ОПТИМИЗАТОР И ФУНКЦИЯ ПОТЕРЬ ==========
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    criterion = nn.L1Loss()  # MAE
    
    # ========== РАННЯЯ ОСТАНОВКА И ЛОГИРОВАНИЕ ==========
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        verbose=True
    )
    logger = Logger(log_dir=config.LOGS_PATH)
    
    # ========== ВОССТАНОВЛЕНИЕ ИЗ ЧЕКПОИНТА ==========
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Восстановлено обучение с эпохи {start_epoch}")
    
    # ========== ЦИКЛ ОБУЧЕНИЯ ==========
    print("\n" + "=" * 60)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 60)
    
    for epoch in range(start_epoch, config.EPOCHS):
        epoch_start_time = time.time()
        
        # Обучение
        train_loss, train_preds, train_targets = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Валидация
        val_loss, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Вычисление метрик
        train_metrics = compute_metrics(train_preds, train_targets)
        val_metrics = compute_metrics(val_preds, val_targets)
        
        # Обновление learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start_time
        
        # Логирование
        logger.add_metrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_mae=train_metrics['mae'],
            val_mae=val_metrics['mae'],
            lr=current_lr,
            epoch_time=epoch_time
        )
        
        # Печать результатов
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        print(f"{'=' * 60}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print_metrics(train_metrics, prefix="Train")
        print_metrics(val_metrics, prefix="Val")
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"Epoch Time: {epoch_time:.2f} сек")
        
        # Ранняя остановка
        if early_stopping(val_loss, epoch, model, config.MODEL_SAVE_PATH):
            print(f"\nEarly stopping triggered! Лучшая модель на эпохе {early_stopping.best_epoch + 1}")
            break
    
    # Сохранение истории обучения
    logger.save()
    save_config(config, os.path.join(config.LOGS_PATH, 'config.json'))
    
    # Загрузка лучшей модели
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"Лучшая эпоха: {logger.get_best_epoch()}")
    print(f"Лучшая val_loss: {logger.get_best_val_loss():.4f}")
    print(f"Лучшая val_MAE: {logger.get_best_val_mae():.4f}")
    print(f"Модель сохранена в: {config.MODEL_SAVE_PATH}")
    
    return model, logger


if __name__ == "__main__":
    # Запуск обучения при прямом вызове файла
    model, logger = train(Config)

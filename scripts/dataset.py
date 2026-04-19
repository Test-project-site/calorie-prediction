"""
Загрузчик данных для проекта.
Обрабатывает изображения и списки ингредиентов.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
from collections import defaultdict

class FoodDataset(Dataset):
    """Датасет для загрузки блюд с изображениями и ингредиентами"""
    
    def __init__(self, csv_path, images_path, split='train', 
                 transform=None, ingredients_transform=None, config=None):
        """
        Args:
            csv_path: путь к dish.csv
            images_path: путь к папке с изображениями
            split: 'train' или 'test'
            transform: аугментации для изображений
            ingredients_transform: трансформации для ингредиентов
            config: объект конфигурации
        """
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data['split'] == split].reset_index(drop=True)
        self.images_path = images_path
        self.transform = transform
        self.ingredients_transform = ingredients_transform
        self.config = config
        
        # Создаём словарь для кодирования ингредиентов
        self._build_ingredient_vocab(csv_path)
        
        # Предварительно парсим все ингредиенты
        self._preprocess_ingredients()
        
    def _build_ingredient_vocab(self, csv_path):
        """Создаёт словарь для преобразования ID ингредиентов в индексы"""
        # Загружаем все ингредиенты
        dish_df = pd.read_csv(csv_path)
        
        # Собираем все уникальные ингредиенты
        all_ingredients = set()
        for ingredients_str in dish_df['ingredients'].dropna():
            for ing_id in ingredients_str.split(';'):
                all_ingredients.add(ing_id)
        
        # Создаём словари для преобразования
        self.ingredient_to_idx = {ing_id: idx for idx, ing_id in enumerate(sorted(all_ingredients))}
        self.idx_to_ingredient = {idx: ing_id for ing_id, idx in self.ingredient_to_idx.items()}
        self.num_ingredients = len(self.ingredient_to_idx)
        
        print(f"Всего уникальных ингредиентов в датасете: {self.num_ingredients}")
        
    def _preprocess_ingredients(self):
        """Предварительно преобразует все списки ингредиентов в индексы"""
        self.ingredients_indices = []
        
        for _, row in self.data.iterrows():
            ingredients_str = row['ingredients']
            if pd.isna(ingredients_str):
                indices = []
            else:
                indices = [self.ingredient_to_idx[ing_id] for ing_id in ingredients_str.split(';')]
            self.ingredients_indices.append(indices)
    
    def _pad_ingredients(self, indices):
        """Дополняет список ингредиентов до MAX_INGREDIENTS"""
        max_len = self.config.MAX_INGREDIENTS if self.config else 30
        
        if len(indices) >= max_len:
            return indices[:max_len]
        else:
            # Padding нулями
            return indices + [0] * (max_len - len(indices))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        dish_id = row['dish_id']
        calories = row['total_calories']
        mass = row['total_mass']
        
        # ========== ЗАГРУЗКА ИЗОБРАЖЕНИЯ ==========
        img_path = os.path.join(self.images_path, dish_id, 'rgb.png')
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Ошибка загрузки изображения {img_path}: {e}")
            # Возвращаем чёрное изображение в случае ошибки
            if self.config:
                image = torch.zeros(3, self.config.IMG_SIZE, self.config.IMG_SIZE)
            else:
                image = torch.zeros(3, 224, 224)
        
        # ========== ОБРАБОТКА ИНГРЕДИЕНТОВ ==========
        ingredients_indices = self.ingredients_indices[idx]
        padded_indices = self._pad_ingredients(ingredients_indices)
        ingredients_tensor = torch.tensor(padded_indices, dtype=torch.long)
        
        # ========== ЦЕЛЕВАЯ ПЕРЕМЕННАЯ ==========
        calories_tensor = torch.tensor(calories, dtype=torch.float32)
        mass_tensor = torch.tensor(mass, dtype=torch.float32)
        
        return {
            'image': image,
            'ingredients': ingredients_tensor,
            'calories': calories_tensor,
            'mass': mass_tensor,
            'dish_id': dish_id,
            'num_ingredients': len(ingredients_indices)
        }


def get_ingredient_embeddings_matrix(ingredient_to_idx, embedding_dim=128):
    """
    Создаёт матрицу для инициализации эмбеддингов ингредиентов.
    Можно использовать предобученные эмбеддинги (например, word2vec),
    но здесь просто случайная инициализация.
    """
    num_ingredients = len(ingredient_to_idx)
    # Случайная инициализация
    embedding_matrix = np.random.randn(num_ingredients, embedding_dim) * 0.01
    # Вектор для паддинга (индекс 0) заполняем нулями
    embedding_matrix[0] = np.zeros(embedding_dim)
    
    return torch.tensor(embedding_matrix, dtype=torch.float32)

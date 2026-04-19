"""
Архитектура нейросети для предсказания калорийности.
Двухпоточная модель: CNN для изображений + Embedding для ингредиентов.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class CaloriePredictor(nn.Module):
    """Модель для предсказания калорийности блюд по изображению и ингредиентам"""
    
    def __init__(self, num_ingredients, embedding_dim=128, 
                 backbone='resnet18', pretrained=True, dropout_rate=0.3):
        """
        Args:
            num_ingredients: количество уникальных ингредиентов
            embedding_dim: размерность эмбеддингов ингредиентов
            backbone: архитектура для извлечения признаков из изображений
            pretrained: использовать ли предобученные веса
            dropout_rate: dropout для регуляризации
        """
        super().__init__()
        
        self.num_ingredients = num_ingredients
        self.embedding_dim = embedding_dim
        
        # ========== ВИЗУАЛЬНЫЙ ПОТОК ==========
        self._init_visual_backbone(backbone, pretrained)
        
        # ========== ТЕКСТОВЫЙ ПОТОК (ингредиенты) ==========
        self.ingredient_embedding = nn.Embedding(
            num_ingredients, 
            embedding_dim,
            padding_idx=0  # индекс 0 - это padding
        )
        
        # Слой для агрегации эмбеддингов ингредиентов
        self.ingredient_pool = nn.AdaptiveAvgPool1d(1)
        
        # ========== ОБЪЕДИНЕНИЕ И РЕГРЕССИЯ ==========
        # Определяем размер признаков после объединения
        visual_features_dim = self._get_visual_features_dim(backbone)
        combined_features_dim = visual_features_dim + embedding_dim
        
        self.regressor = nn.Sequential(
            nn.Linear(combined_features_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 1)
        )
        
    def _init_visual_backbone(self, backbone, pretrained):
        """Инициализирует backbone для изображений"""
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            # Заменяем последний слой на Identity
            self.backbone.fc = nn.Identity()
            
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            self.backbone.fc = nn.Identity()
            
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            self.backbone.fc = nn.Identity()
            
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Заморозим первые слои при использовании предобученной модели
        if pretrained:
            self._freeze_backbone_layers()
    
    def _freeze_backbone_layers(self):
        """Замораживает первые слои backbone, чтобы они не обучались"""
        # Замораживаем все слои
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Размораживаем последние 2 блока (для fine-tuning)
        # Для ResNet18/34: layer3 и layer4
        if hasattr(self.backbone, 'layer3'):
            for param in self.backbone.layer3.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, 'layer4'):
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
        
    def _get_visual_features_dim(self, backbone):
        """Возвращает размерность выходных признаков backbone"""
        if backbone in ['resnet18', 'resnet34']:
            return 512
        elif backbone == 'resnet50':
            return 2048
        elif backbone == 'efficientnet_b0':
            return 1280
        else:
            return 512
    
    def forward(self, image, ingredients):
        """
        Args:
            image: тензор изображения [batch, 3, H, W]
            ingredients: тензор ингредиентов [batch, max_ingredients]
        
        Returns:
            предсказанные калории [batch]
        """
        # ========== ВИЗУАЛЬНЫЕ ПРИЗНАКИ ==========
        visual_features = self.backbone(image)  # [batch, visual_dim]
        
        # ========== ПРИЗНАКИ ИНГРЕДИЕНТОВ ==========
        # Получаем эмбеддинги для всех ингредиентов
        ing_embeddings = self.ingredient_embedding(ingredients)  # [batch, seq_len, emb_dim]
        
        # Агрегируем по последовательности (среднее по всем ингредиентам)
        # Создаём маску для игнорирования паддинга
        mask = (ingredients != 0).float().unsqueeze(-1)  # [batch, seq_len, 1]
        ing_embeddings_masked = ing_embeddings * mask
        ing_features = ing_embeddings_masked.sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # [batch, emb_dim]
        
        # ========== ОБЪЕДИНЕНИЕ ==========
        combined = torch.cat([visual_features, ing_features], dim=1)  # [batch, visual_dim + emb_dim]
        
        # ========== РЕГРЕССИЯ ==========
        calories = self.regressor(combined).squeeze()  # [batch]
        
        return calories
    
    def unfreeze_all_layers(self):
        """Размораживает все слои для полного fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True

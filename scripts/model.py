import torch
import torch.nn as nn
import torchvision.models as models

class CaloriePredictor(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True):
        super().__init__()
        
        # Используем предобученную модель
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Убираем последний слой
        
        # Свои слои для регрессии
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Одно число — калории
        )
        
    def forward(self, x):
        features = self.backbone(x)
        calories = self.regressor(features)
        return calories.squeeze()

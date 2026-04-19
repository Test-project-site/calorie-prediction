import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
import os

class FoodDataset(Dataset):
    def __init__(self, csv_path, images_dir, split='train', transform=None):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data['split'] == split]
        self.images_dir = images_dir
        self.transform = transform or transforms.ToTensor()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        dish_id = row['dish_id']
        calories = row['total_calories']
        
        # Загрузка изображения
        img_path = os.path.join(self.images_dir, str(dish_id), 'rgb.png')
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # TODO: Добавить обработку ингредиентов
        
        return {
            'image': image,
            'calories': torch.tensor(calories, dtype=torch.float32),
            'dish_id': dish_id
        }

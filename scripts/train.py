import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    for batch in tqdm(dataloader, desc='Training'):
        images = batch['image'].to(device)
        calories = batch['calories'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, calories)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions.extend(outputs.cpu().detach().numpy())
        targets.extend(calories.cpu().numpy())
    
    return total_loss / len(dataloader), predictions, targets

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            images = batch['image'].to(device)
            calories = batch['calories'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, calories)
            
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            targets.extend(calories.cpu().numpy())
    
    return total_loss / len(dataloader), predictions, targets

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET



model = NeuralNetwork().to(device)
criterion = nn.L1Loss() # You can choose another loss function depending on your problem
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(model, dataloader, criterion, optimizer, device):  
    model.train()  
    running_loss = 0.0  
    for batch in dataloader:  
        inputs = batch['image'].to(device)  
        targets = batch['bboxes'].to(device)  
        
        optimizer.zero_grad()  
        outputs = model(inputs)  
        loss = criterion(outputs, targets)  
        loss.backward()   
        optimizer.step()  
        
        running_loss += loss.item()  
    return running_loss / len(dataloader)  



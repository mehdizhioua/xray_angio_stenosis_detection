import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET


class ObjectDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.backbone = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes + 1),
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4),
        )
        
    def forward(self, x):
        features = self.backbone(x)
        flattened = nn.Flatten()(features)
        classes = self.classifier(flattened)
        regressor = self.regressor(flattened)
        
        return classes, regressor



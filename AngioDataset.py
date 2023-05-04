import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET

class ObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.image_names = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        annotation_path = os.path.join(self.annotation_dir, image_name.replace('.jpg', '.xml'))
        
        image = Image.open(image_path).convert('RGB')
        
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        boxes = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, boxes

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
from Model import StenosisDetector
from AngioDataset import MyDataset



backbone = "Faster RCNN Resnet 50"
detector = StenosisDetector(backbone)
detector.load_model()

FRCNN = detector.model

if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
    print('GPU not available')

#uncomment if working on nvidia
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(device)

img_dir = 'sample_data/image_dir'
xml_dir = 'sample_data/xml_dir'
dataset = MyDataset(img_dir=img_dir, xml_dir=xml_dir)
angio_data = torch.utils.data.DataLoader(dataset)
FRCNN.to(device)

angio_iter = iter(angio_data)

# Get a single batch of data
img = next(angio_iter)[1]

out = FRCNN(img)

print(out)




#criterion = nn.L1Loss() # You can choose another loss function depending on your problem
#optimizer = optim.Adam(model.parameters(), lr=0.001)


#def train(model, dataloader, criterion, optimizer, device):  
#    model.train()  
#    running_loss = 0.0  
#    for batch in dataloader:  
#        inputs = batch['image'].to(device)  
#        targets = batch['bboxes'].to(device)  
#        
#        optimizer.zero_grad()  
#        outputs = model(inputs)  
#        loss = criterion(outputs, targets)  
#        loss.backward()   
#        optimizer.step()  
        
#        running_loss += loss.item()  ,
#    return running_loss / len(dataloader)  



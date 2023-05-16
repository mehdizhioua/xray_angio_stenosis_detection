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
import torchvision


#backbone = "Faster RCNN Resnet 50"
#detector = StenosisDetector(backbone)
#detector.load_model()

#FRCNN = detector.model


#uncomment if working on nvidia
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



#img_dir = 'sample_data/image_dir'
#xml_dir = 'sample_data/xml_dir'
#dataset = MyDataset(img_dir=img_dir, xml_dir=xml_dir)
#angio_data = torch.utils.data.DataLoader(dataset)

#angio_iter = iter(angio_data)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# For training
images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
labels = torch.randint(1, 91, (4, 11))
images = list(image for image in images)
targets = []
for i in range(len(images)):
    d = {}
    d['boxes'] = boxes[i]
    d['labels'] = labels[i]
    targets.append(d)
output = model(images, targets)
print(output)

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



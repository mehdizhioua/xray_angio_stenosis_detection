import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
from Model import StenosisDetector
from AngioDataset import MyDataset
from utils import *
import torchvision


backbone = "Faster RCNN Resnet 50"
detector = StenosisDetector(backbone)
detector.load_model()
FRCNN = detector.model
img_dir = 'sample_data/image_dir'
xml_dir = 'sample_data/xml_dir'
dataset = MyDataset(img_dir=img_dir, xml_dir=xml_dir)
angio_data = torch.utils.data.DataLoader(dataset,batch_size=2)

_, input_try, box_try = next(iter(angio_data))

images,targets = preprocess(input_try,box_try)

FRCNN.eval()
output = FRCNN(images,)




# Define the device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the optimizer
#optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Number of training epochs
#num_epochs = 25

# Define the dataset loader
#data_loader = torch.utils.data.DataLoader(
#    your_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=your_collate_fn
#)

#for epoch in range(num_epochs):
#    print(f"Epoch: {epoch}/{num_epochs}")

#    for i, data in enumerate(data_loader):
#        # Get the inputs and labels
#        images, targets = data

        # Transfer them to the device
#        images = list(image.to(device) for image in images)
#        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Zero the parameter gradients
#        optimizer.zero_grad()

        # Forward pass
#        loss_dict = model(images, targets)

        # Calculate total loss
#        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
#        losses.backward()

        # Perform a single optimization step
#        optimizer.step()

#        if i % 10 == 0:
#            print(f"Iteration: {i}, Loss: {losses.item()}")

#print("Finished Training")

#print(len(output))

#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# For training
#images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)


#boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
#labels = torch.randint(1, 91, (4, 11))
#images = list(image for image in images)
#targets = []
#for i in range(len(images)):
#    d = {}
#    d['boxes'] = boxes[i]
#    d['labels'] = labels[i]
#    targets.append(d)
#output = model(images, targets)
#print(output)

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



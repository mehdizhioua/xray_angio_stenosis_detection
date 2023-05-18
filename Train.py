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


#load the model
backbone = "Faster RCNN Resnet 50"
detector = StenosisDetector(backbone)
detector.load_model()
FRCNN = detector.model

#load the data
img_dir = 'sample_data/image_dir'
xml_dir = 'sample_data/xml_dir'
dataset = MyDataset(img_dir=img_dir, xml_dir=xml_dir)
angio_data = torch.utils.data.DataLoader(dataset,batch_size=2)


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FRCNN.to(device)

# Define the optimizer
optimizer = optim.SGD(FRCNN.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

print("starting the training")

#Training loop
num_epochs = 1
for epoch in range(num_epochs):
    print(epoch)
    for i, data in enumerate(angio_data):
        _, inpt, label = data
        images,targets = preprocess(inpt,label)
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        loss_dict = FRCNN(images, targets)
        print(loss_dict)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
   
        print(losses.item())

print("Finished Training")


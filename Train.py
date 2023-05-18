from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
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


#plt.ioff()

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

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    print("Epoch:", epoch)
    for i, data in enumerate(angio_data):
        name = 'image_storage/test' + str(i) + '.png'
        _, inpt, label = data
        points = label[0]
        plt.figure()
        plt.imshow(inpt[0,:,:,:])  # the first dimension is of size 2, but i wasn't sure whether to go with 0 or 1, as 0 looked fine
        plt.scatter([points[0][0].item(), points[1][0].item(), points[2][0].item(), points[3][0].item()], [points[0][1].item(), points[1][1].item(), points[2][1].item(), points[3][1].item()])
        plt.savefig(name)
        images,targets = preprocess(inpt,label)
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        
        loss_dict = FRCNN(images, targets)
        print('Classifier_Loss:', loss_dict['loss_classifier'].item(), 'BBox_Reg_Loss:', loss_dict['loss_box_reg'].item(), 'RPN_BBox_Reg_Loss:', loss_dict['loss_rpn_box_reg'].item(), 'Objectness_Loss:', loss_dict['loss_objectness'].item())
        
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
   
        print('Net Loss', losses.item())

print("Finished Training")

    
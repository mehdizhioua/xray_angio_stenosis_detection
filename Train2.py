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
import numpy as np
from Model import StenosisDetector
from torch.utils.tensorboard import SummaryWriter
from AngioDataset import *
from utils import *
import torchvision
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


#setting GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


#loading the data
img_dir = 'data/image_dir'
xml_dir = 'data/xml_dir'
train_loader, dev_loader, test_loader = get_train_dev_test(img_dir, xml_dir, batch=7, dev_split=0.15, test_split=0.15)

print("--------DATA LOADED--------")
#loading the model
backbone = "Faster RCNN Resnet 50"
detector = StenosisDetector(backbone)
detector.load_model()
FRCNN = detector.model
FRCNN.to(device)

optimizer = optim.SGD(FRCNN.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

writer = SummaryWriter("runs/may_31")

def run_epochs(num_epochs, model, train_data, dev_data, total_batches_=0):
    """
    Does the Training loop below but with multiple epochs? Will need further modification.
    """
    for epoch in range(num_epochs):
        # Training step
        model.train()
        for i, data in enumerate(train_data):
            inpt, label = data
            img_i = preprocess_img(inpt,device)
            target_i = preprocess_label(label,device)
            if i%300==0 and i>0:
                visualize(FRCNN,img_i,target_i,"train_"+str(i)+"_epoch_"+str(n_epoch)+".png")
            optimizer.zero_grad()
            loss_dict = model(img_i, target_i)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            
            #keep track of losses 
            writer.add_scalar("Train/Loss", losses.item(), total_batches_)
            writer.add_scalar("Train/Classifier_Loss", loss_dict['loss_classifier'].item(), total_batches_)
            writer.add_scalar("Train/BBox_Reg_Loss",loss_dict['loss_box_reg'].item(), total_batches_)
            writer.add_scalar("Train/RPN_BBox_Reg_Loss", loss_dict['loss_rpn_box_reg'].item(), total_batches_)
            writer.add_scalar("Train/Objectness_Loss",loss_dict['loss_objectness'].item(), total_batches_)
            total_batches_+=1

        # dev set
        with torch.no_grad():
            total_loss = 0
            for i, data in enumerate(dev_data):
                inpt, label = data
                img_i = preprocess_img(inpt,device)
                target_i = preprocess_label(label,device)
                loss_dict = model(img_i, target_i)
                total_loss += sum(loss for loss in loss_dict.values()).item()
            avg_loss = total_loss / len(dev_data)
            writer.add_scalar("Dev/Loss", avg_loss, epoch)

    return total_batches_



def train(N_epoch, train_data, dev_data):
    total_batches = 0
    print("Device",device)
    print("starting the training")
    for epch in range(N_epoch):
        print("epoch",epch)
        inpt_e, label_e = next(iter(train_data))
        img_e = preprocess_img(inpt_e,device)
        target_e = preprocess_label(label_e,device)
        visualize(FRCNN,img_e,target_e,"test"+str(epch)+".png")
        total_btches = run_epochs(1, FRCNN, train_data, dev_data, total_batches)
        total_batches+=total_btches
    print("Finished training")

    
writer.close()

train(3, train_loader, dev_loader)

#save the model    
torch.save(FRCNN, "saved_models/fast_rcnn_resnet_may31.pth")
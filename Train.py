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
img_dir = 'big_data/image_dir'
xml_dir = 'big_data/xml_dir'
angio_data = get_data_loader(img_dir,xml_dir,batch=7)


print("DATA LOADED")



#loading the model
backbone = "Faster RCNN Resnet 50"
detector = StenosisDetector(backbone)
detector.load_model()
FRCNN = detector.model
FRCNN.to(device)

optimizer = optim.SGD(FRCNN.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

writer = SummaryWriter("runs/may_19_10am")

def run_epochs(num_epochs,writer_,model,n_epoch,total_batches_=0):
    """
    Does the Training loop below but with multiple epochs? Will need further modification.
    """
    for epoch in range(num_epochs):
        for i, data in enumerate(angio_data):
            inpt, label = data
            img_i = preprocess_img(inpt,device)
            target_i = preprocess_label(label,device)
            if i%300==0 and i>0:
                visualize(FRCNN,img_i,target_i,"train_"+str(i)+"_epoch_"+str(n_epoch)+".png")
            optimizer.zero_grad()
            loss_dict = FRCNN(img_i, target_i)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            
            #keep track of losses 
            writer_.add_scalar("Loss/train", losses.item(), total_batches_)
            writer_.add_scalar("Classifier_Loss/train", loss_dict['loss_classifier'].item(), total_batches_)
            writer_.add_scalar("BBox_Reg_Loss/train",loss_dict['loss_box_reg'].item(), total_batches_)
            writer_.add_scalar("RPN_BBox_Reg_Loss/train", loss_dict['loss_rpn_box_reg'].item(), total_batches_)
            writer_.add_scalar("Objectness_Loss/train",loss_dict['loss_objectness'].item(), total_batches_)
            total_batches_+=1
            
    return total_batches_   


def train(N_epoch):
    total_batches = 0
    print("Device",device)
    print("starting the training")
    for epch in range(N_epoch):
        print("epoch",epch)
        inpt_e, label_e = next(iter(angio_data))
        img_e = preprocess_img(inpt_e,device)
        target_e = preprocess_label(label_e,device)
        visualize(FRCNN,img_e,target_e,"test"+str(epch)+".png")
        total_btches = run_epochs(1,writer,FRCNN,epch,total_batches)
        total_batches+=total_btches
    print("Finished training")

    
    
writer.close()

train(5)

#save the model    
torch.save(FRCNN, "saved_models/fast_rcnn_resnet_50_5_epch.pth")


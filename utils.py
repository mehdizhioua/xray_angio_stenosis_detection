
import torch
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
from AngioDataset import AngioData
import torchvision
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

def np_to_torch(inpt_try):
    return (torch.from_numpy(np.transpose(inpt_try, (2, 0, 1))).float()/255.0)

def torch_to_np(inpt_try):
    inpt_try_np = inpt_try.cpu().numpy()
    return np.transpose(inpt_try_np, (1, 2, 0))

def preprocess_img(input_try_,device_):
    return [np_to_torch(inpt_try).to(device_) for inpt_try in input_try_]

def preprocess_label(label_try_,device_):
    return [{k: v.to(device_) for k, v in t.items()} for t in label_try_]
        
def visualize(model_,img_try_,target_try_,name="test_test.png"):
    """
    img_try_ and target_try_ in tensor format as they should be 
    fed in the model
    """
    
    model_.eval()
    pred = model_(img_try_)[0]
    true = target_try_[0]
    path = "image_storage/"+name
    
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    
    img_numpy = torch_to_np(img_try_[0]) 
    
    ax.imshow(img_numpy)
    
    #plot the predicted patches
    for box in pred["boxes"]:
        xy_min_max = np.array(box.cpu().detach())
        xmin,ymin = xy_min_max[0], xy_min_max[1]
        xmax,ymax = xy_min_max[2], xy_min_max[3]
        width = xmax - xmin
        height = ymax - ymin
        rect = Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
    #plot the true patches
    if "boxes" in true and len(true["boxes"])>0:
        for box in true["boxes"]:
            xy_min_max = np.array(box.cpu().detach())
            xmin,ymin = xy_min_max[0], xy_min_max[1]
            xmax,ymax = xy_min_max[2], xy_min_max[3]
            width = xmax - xmin
            height = ymax - ymin
            rect = Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
        
    plt.savefig(path)
    #go back to training mode after visualization
    model_.train()


import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import xml.etree.ElementTree as ET

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader
import numpy as np


class StenosisDetector():
    """
    attributes :

    - model 
    - backbone
    """
    def __init__(self,backbn):
        self.backbone = backbn
        self.load_model()

    def load_model(self):
        if self.backbone=="Faster RCNN Resnet 50":
            model = fasterrcnn_resnet50_fpn(pretrained=True)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
            self.model = model


backbone = "Faster RCNN Resnet 50"


detector = StenosisDetector(backbone)
detector.load_model()
print(detector.model)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

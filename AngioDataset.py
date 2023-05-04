import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import xml.etree.ElementTree as ET
import os
import cv2


def parse_annotation(xml_file):  
    tree = ET.parse(xml_file)  
    root = tree.getroot()  
    bboxes = []   
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        bboxes.append([int(bbox.find('xmin').text), int(bbox.find('ymin').text),
                       int(bbox.find('xmax').text), int(bbox.find('ymax').text)])
    return bboxes


class CustomDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transform=None):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.img_filenames = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_filenames[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        annotation_path = os.path.join(self.annotation_dir, self.img_filenames[idx].split('.')[0] + '.xml')
        bboxes = parse_annotation(annotation_path)
        
        sample = {'image': img, 'bboxes': bboxes}
        if self.transform:
            sample = self.transform(sample)
        return sample  


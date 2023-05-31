import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import xml.etree.ElementTree as ET
import os
import cv2


def parse_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        bboxes.append([int(bbox.find('xmin').text), int(bbox.find('ymin').text),
                       int(bbox.find('xmax').text), int(bbox.find('ymax').text)])
    return {"boxes":bboxes,"labels":[1]*len(bboxes)}

class AngioData(torch.utils.data.Dataset):
    def __init__(self, img_dir, xml_dir):
        self.image_dir = img_dir
        self.xml_dir = xml_dir
        self.image_list = os.listdir(img_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # load image
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_name)
        images = cv2.imread(image_path)

        # parse the XML file and extract the desired information        
        xml_path = os.path.join(self.xml_dir, image_name.replace('.bmp', '.xml'))
        bboxes = parse_annotation(xml_path)
        
        return image_name, images, bboxes

def collate_fn(batch):
    images = []
    targets = []
    for sample in batch:
        images.append(sample[1])
        targets.append({'boxes': torch.tensor(sample[2]['boxes']), 'labels': torch.tensor(sample[2]['labels'])})
    return images, targets


def get_data_loader(img_path,ann_path,batch=3):
    dataset = AngioData(img_dir=img_path, xml_dir=ann_path)
    data_loader = torch.utils.data.DataLoader(dataset,batch_size=batch,collate_fn=collate_fn)
    return data_loader



#def parse_annotation(xml_path):
#    tree = ET.parse(xml_path)
#    root = tree.getroot()
#    bboxes = []
#    for obj in root.findall('object'):
#        bbox = obj.find('bndbox')
#        bboxes.append([int(bbox.find('xmin').text), int(bbox.find('ymin').text),
#                       int(bbox.find('xmax').text), int(bbox.find('ymax').text)])
#    return bboxes

#class MyDataset(torch.utils.data.Dataset):
#    def __init__(self, img_dir, xml_dir):
#        self.image_dir = img_dir
#        self.xml_dir = xml_dir
#        self.image_list = os.listdir(img_dir)

#    def __len__(self):
#        return len(self.image_list)

#    def __getitem__(self, idx):
#        # load image
#        image_name = self.image_list[idx]
#        image_path = os.path.join(self.image_dir, image_name)
#        images = cv2.imread(image_path)

#        # parse the XML file and extract the desired information        
#        xml_path = os.path.join(self.xml_dir, image_name.replace('.bmp', '.xml'))
#        bboxes = parse_annotation(xml_path)
        
#        return image_name, images, bboxes
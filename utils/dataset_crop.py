import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import os
import cv2
import numpy as np
from torchvision import datasets, models, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import FancyPCA
class roadDataset(Dataset):
    def __init__(self, data_dir,is_train=True):
        self.paths=sorted(glob.glob(data_dir+'/*/*'))
        self.transform_train = A.Compose([
            # A.Resize(height=700, width=1000),
            A.RandomResizedCrop(height=600, width=500),  # change ratio and scale 500 / 900 
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.RandomGamma(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10,p=0.5),
            A.OneOf([A.MotionBlur(blur_limit=3), A.GlassBlur(max_delta=3), A.GaussianBlur(blur_limit=3)], p=0.5),
            A.GaussNoise(p=0.5),
            A.Normalize(mean=(0.446, 0.469, 0.472), std=(0.326, 0.330, 0.338), max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ])

        self.transform_valid = A.Compose([
            # A.Resize(height=image_size, width=image_size),
            A.Resize(height=600, width=500),
            # A.RandomResizedCrop(height=700, width=1000),
            A.Normalize(mean=(0.446, 0.469, 0.472), std=(0.326, 0.330, 0.338), max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ])
        if is_train:
            self.data_transforms=self.transform_train
        else:
            self.data_transforms = self.transform_valid

    def __getitem__(self,index):
        #
        sample_path = self.paths[index]
        cls = sample_path.split('/')[-2]
        label = int(cls)
        img=Image.open(sample_path)
        img = img.convert('RGB')
        img=np.array(img)
        img = self.data_transforms(image=img)['image']

        return img,label

    def __len__(self):
        return len(self.paths)
class roadDatasetInfer(Dataset):
    def __init__(self, data_dir):
        self.paths=sorted(glob.glob(data_dir+'/*/*'))
        self.data_transforms = A.Compose([
            A.Resize(height=600, width=500),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.446, 0.469, 0.472), std=(0.326, 0.330, 0.338), max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ])

    def __getitem__(self,index):

        sample_path = self.paths[index]
        img=Image.open(sample_path)
        img = img.convert('RGB')
        img=np.array(img)
        img = self.data_transforms(image=img)['image']
        return img,sample_path

    def __len__(self):
        return len(self.paths)



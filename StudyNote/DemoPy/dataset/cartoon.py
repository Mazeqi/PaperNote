import os
import glob
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image

class Cartoon(Dataset):

    def __init__(self, transform=None):
        
        self.transform = transform
        
        self.img_files= sorted(glob.glob('../data/cartoon/' + '*.jpg'))
        #print(self.img_files)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img = self.load_img(index)

        if self.transform:
            sample = self.transform(img)

        return sample

    
    def load_img(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img =  Image.open(img_path)
        
        return img
    
training_set = Cartoon(transform=transforms.Compose([transforms.Resize(28), 
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.5], [0.5])
                                                    ])
                      )

training_generator = DataLoader(training_set)

for i_iter, data in enumerate(training_generator):
        print(data)
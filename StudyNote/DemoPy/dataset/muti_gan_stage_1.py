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

class defect_patch(Dataset):

    def __init__(self, transform = None, label_bg_path = ['crop_img_catagory/3_background/black.jpg', 'crop_img_catagory/3_background/blue.jpg']):
        
        self.transform = transform
        self.label_bg_path = label_bg_path
        
        self.img_files_0 = glob.glob('crop_img_catagory/3_black/' + '*.jpg')
        self.img_files_1 = glob.glob('crop_img_catagory/3_blue/' + '*.jpg')

        self.label_0 = ['black' for i in range(len(self.img_files_0))]
        self.label_1 = ['blue' for i in range(len(self.img_files_1))]

        self.label_0.extend(self.label_1)
        self.label = self.label_0

        self.img_files_0.extend(self.img_files_1)
        self.img_files = self.img_files_0


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img = self.load_img(index)
        if self.transform:
            img = self.transform(img)
        
        label_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        label_i = self.label[index % len(self.label)]
        label_img = None

        if label_i == 'black':
            label_img = Image.open(self.label_bg_path[0])
        else:
            label_img = Image.open(self.label_bg_path[1])

        label_img = label_transform(label_img)

        return img, label_img

    def load_img(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img =  Image.open(img_path)

        return img

# DataLoader中collate_fn使用
def collate(batch):
    images = []
    labels = []
    for img, lab in batch:
        images.append(img)
        labels.append(lab)
    images = np.array(images, dtype = object)
    labels = np.array(labels)
    return images, labels

if __name__ == "__main__":
    training_set = defect_patch(transform=transforms.Compose([transforms.Resize([140, 20]), 
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.5], [0.5])
                                                    ])
                                )

    dataloader = DataLoader(training_set, batch_size = 32, shuffle=True)

    for img, label_img in dataloader:
        np_img = np.array(label_img[0].squeeze().permute([1,2,0]))
        print(np_img.shape)
        cv2.namedWindow('test', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('test',np_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
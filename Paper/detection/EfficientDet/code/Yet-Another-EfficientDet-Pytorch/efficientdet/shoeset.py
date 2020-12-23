from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np
import cv2
import argparse
import yaml
from torchvision import transforms
import tqdm

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


class ShoeSet(Dataset):

    def __init__(self, list_path = "VOC2020.02/train.txt", transform=None):
        
        self.transform = transform

        with open(list_path, "r") as file:
            self.img_files = file.readlines()
        
        

        self.label_files = [
            path.replace("JPEGImages", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
    

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img = self.load_img(index)
        annotations = self.load_annotations(index)

        cv2.namedWindow('test1', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('test1',img )
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        


        sample = {'img': img, 'annot':annotations}

        if self.transform:
            sample = self.transform(sample)

        return sample

    
    def load_img(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        #print(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img.astype(np.float32) / 255.
    

    def load_annotations(self, index):

        label_path = self.label_files[index % len(self.img_files)].rstrip()
        annotations = None

        # id x1 y1 x2 y2
        if os.path.exists(label_path):
            i_label = np.loadtxt(label_path).reshape(-1,5)
            annotations = np.zeros(np.shape(i_label))
            
            annotations[:, 0] = i_label[:, 1]
            annotations[:, 1] = i_label[:, 2]
            annotations[:, 2] = i_label[:, 3]
            annotations[:, 3] = i_label[:, 4]
            annotations[:, 4] = i_label[:, 0]

        
        return annotations


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}


if __name__ == "__main__":

    params = Params('projects/shoe.yml')

    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
    parser.add_argument('-n', '--num_workers', type=int, default=0, help='num_workers of dataloader')

    args = parser.parse_args()
    

    training_params = {'batch_size': args.batch_size,
                    'shuffle': True,
                    'drop_last': True,
                    'collate_fn': collater,
                    'num_workers': args.num_workers}


    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

    training_set = ShoeSet(transform=transforms.Compose([#Normalizer(mean=params.mean, std=params.std),
                                                         #Augmenter(),
                                                         Resizer(input_sizes[args.compound_coef])]) )

    training_generator = DataLoader(training_set,**training_params)

    for iter, data in enumerate(training_generator):
        #print(np.shape(data['img']))
        img = data['img'][0]
        img = img.squeeze()
        img = img.permute(1,2,0)
        img = img.numpy()
        cv2.namedWindow('test', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('test',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        







    



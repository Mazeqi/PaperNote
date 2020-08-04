from torch import nn
import torch
import torch.utils.data
from torch.nn import functional as F

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules import Module

class conv1d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size = 3, stride=1):
        super(conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        #self.padding = padding
        self.kernel_size = kernel_size

    def forward(self, img):
        input_rows  = img.size(2)

        output_rows = (input_rows + self.stride - 1) // self.stride

        padding_rows = max(0, (output_rows - 1) * self.stride + self.kernel_size - input_rows)

        self.conv =  nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size,  stride = self.stride, padding = padding_rows//2)
        conv.to('cuda')
        out = self.conv(img)

        return out

   

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super(Block, self).__init__()
        self.conv1 = nn.Sequential(
            conv1d(in_channels, out_channels, kernel_size),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),

            conv1d(out_channels, out_channels, kernel_size),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),

            conv1d(out_channels, 1, kernel_size),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            conv1d(in_channels, out_channels, kernel_size = 1)
        )

    def forward(self, img):
        #img = img.reshape(img.shape[0], img.shape[1], -1)
        out_1 = self.conv1(img)
        out_2 = self.conv2(img)

        out_total = torch.add(out_1, out_2)
        return out_total

class Block2(nn.Module):
    def __init__(self,in_channels, out_channels = 64, kernel_size = 5):
        super(Block2, self).__init__()
        self.conv1 = nn.Sequential(
            Block(in_channels, out_channels * 2, kernel_size),
            nn.MaxPool1d(kernel_size = 2),
            nn.Dropout(0.3),

            Block(out_channels*2, out_channels, kernel_size),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, img):
        out = self.conv1(img)
        return out

class Model(nn.Module):
    def __init__(self, num_classes = 19):
        super(Model, self).__init__()
        self.B1 = nn.Sequential(
            Block2(in_channels = 1, out_channels = 64, kernel_size = 3)
        )

        self.B2 = nn.Sequential(
            Block2(in_channels = 1, out_channels = 64, kernel_size = 5)
        )

        self.B3 = nn.Sequential(
            Block2(in_channels = 1, out_channels = 64, kernel_size = 7)
        )

        self.dense = nn.Sequential(
            nn.Linear(in_features = 192, out_features = 512),
            nn.ReLU(inplace = True),
            nn.Dropout(0.3),

            nn.Linear(in_features = 512, out_features = 128),
            nn.ReLU(inplace = True),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Linear( in_features = 128, out_features = 19)
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                # nn.init.constant_(m.weight, 0)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, img):
        img = img.reshape(img.shape[0], img.shape[1], -1)
        out_1 = self.B1(img)
        out_2 = self.B2(img)
        out_3 = self.B3(img)
        #print(out_1.size())
        out_total = torch.cat([out_1, out_2, out_3], dim = 1)
        out_total = torch.squeeze(out_total)
        #out_total = out_total.reshape(img.shape[0], 192)

        out = self.dense(out_total)

        out = self.classifier(out)

        return out


        
if __name__ == "__main__":
    
    net = Model(19)
    y = (torch.randn(2, 1, 60, 8))
    out = net(y)
    print(out.size())

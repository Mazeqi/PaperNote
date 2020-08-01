from torch import nn
import torch.nn.functional as F
import torch
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1,  bias=False,)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride = 1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()

        if stride != 1 or inplanes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion * planes)
            )

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, img):

        out = self.conv1(img)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(img)
        out = self.relu(out)

        return out


# ResNet18除了第一块的每个块的第一层的stride为2，非第一层为1
# 第一块的stride为1
class ResNet18(nn.Module):
    def __init__(self, num_class=10):
        super(ResNet18, self).__init__()
        self.inplanes = 16
        self.avg_pool =  nn.AdaptiveAvgPool2d((1,1))
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlock, 16, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 2, stride=2)
        self.linear = nn.Linear(64 * BasicBlock.expansion, num_class)

    def _make_layer(self, block, planes, num_block,stride=1):
        layers=[]
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, img):
       out = self.conv1(img)
       out = self.bn1(out)
       out = F.relu(out)

       out = self.layer1(out)
       out = self.layer2(out)
       out = self.layer3(out)
       out = self.avg_pool(out)
       out = torch.flatten(out, 1)
       out = self.linear(out)

       return out

if __name__ == "__main__":
    net = ResNet18(10)
    y = net((torch.randn(1, 3, 60, 8)))
    print(y)

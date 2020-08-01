from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

def conv3x3(in_planes, out_planes, stride = 1, kernel_size = 3):
    return nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, padding = 1, bias = False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride = 1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, out_planes, stride = stride)
        self.bn1   = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace = True)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion * out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * out_planes, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(self.expansion * out_planes)
            )
        
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2   = nn.BatchNorm2d(out_planes)


    def forward(self, img):
        out = self.conv1(img)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(img)
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, num_class = 19):
        super(ResNet18, self).__init__()

        self.in_planes = 16
        self.avg_pool  = nn.AdaptiveAvgPool2d((1,1))
        self.conv1     = conv3x3(3, 16)
        self.bn1       = nn.BatchNorm2d(16)

        self.layer1    = self._make_layer(BasicBlock, out_planes = 16, num_block = 2, stride = 1)
        self.layer2    = self._make_layer(BasicBlock, out_planes = 32, num_block = 2, stride = 2)
        self.layer3    = self._make_layer(BasicBlock, out_planes = 64, num_block = 2, stride = 2)
        self.last_conv = nn.Conv2d(in_channels = BasicBlock.expansion* 64, out_channels = num_class, kernel_size = 1)
        self._initialize_weights()
        #self.last_conv = nn.Linear(64 * BasicBlock.expansion, num_class)
    
    def _make_layer(self, block, out_planes, num_block, stride = 1):
        layers = []
        layers.append(block(self.in_planes, out_planes, stride = stride))
        self.in_planes = out_planes * block.expansion

        for _ in range(1, num_block):
            layers.append(block(self.in_planes, out_planes))
        
        return nn.Sequential(*layers)
    

    def forward(self, img):
        out = self.conv1(img)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
       
        out = self.last_conv(out)
        # print(out)
        out = torch.flatten(out, 1)
        # print(out)
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv2d):
                # nn.init.constant_(m.weight, 0)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    net = ResNet18(19)
    y = net((torch.randn(1, 3, 60, 8)))
    print(y.size())



    

from torch import nn
import torch

class Model(nn.Module):
    def __init__(self, num_classes=19):
        super(Model, self).__init__()
        # input: 1, num, features_num
        base_channel=64
        self.conv1 = nn.Sequential(
           nn.Conv1d(1, 64, 3, stride=1, padding = 1), 
           nn.BatchNorm1d(base_channel), 
           nn.ReLU(inplace=True),

           nn.Conv1d(base_channel, base_channel * 2, 3, stride=1, padding = 1), 
           nn.BatchNorm1d(base_channel * 2), 
           nn.ReLU(inplace=True),

           nn.Conv1d(base_channel * 2, base_channel * 4, 3, stride=1, padding = 1), 
           nn.BatchNorm1d(base_channel * 4), 
           nn.ReLU(inplace=True),
        )
       
        self.conv2 = nn.Sequential(
            # 1
            nn.Conv2d(base_channel * 4, base_channel * 4, kernel_size=(3, 3),stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(base_channel * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(base_channel * 4, base_channel * 2,  kernel_size=(1, 1)),
            nn.BatchNorm2d(base_channel * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channel * 2, base_channel,  kernel_size=(1, 1)),
            nn.BatchNorm2d(base_channel),
            nn.ReLU(inplace=True),
            # 2
            nn.Conv2d(base_channel, base_channel*2, kernel_size=(3, 3), stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(base_channel*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(base_channel * 2, base_channel*4,kernel_size=(3, 3), stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(base_channel*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            # last
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Dropout(0.3),
        )
        self.classier = nn.Linear(base_channel*4, num_classes)
        self._initialize_weights()

    def forward(self, img):
        
        base_channel=64
        x = img.reshape(img.shape[0], img.shape[1], -1)

        x = self.conv1(x)
        x = x.reshape(img.shape[0], base_channel * 4, 60, 8)

        x = self.conv2(x)
        x = x.view(img.shape[0], -1)
        x = self.classier(x)
        return x

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
    net = Model(19)
    y = net((torch.randn(20, 1, 60, 8)))
    print(y.size())
    #m = nn.Conv2d(1, 33, kernel_size=(1, 1))
    #into = torch.randn(20, 1, 60, 8)
    #output = m(into)
    #print(output.size())

   
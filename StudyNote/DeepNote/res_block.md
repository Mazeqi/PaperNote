- [参考1](https://zhuanlan.zhihu.com/p/42833949) [参考2](https://zhuanlan.zhihu.com/p/42706477)
- 在统计学中，残差和误差是非常容易混淆的两个概念。误差是衡量观测值和真实值之间的差距，残差是指预测值和观测值之间的差距。对于残差网络的命名原因，作者给出的解释是，网络的一层通常可以看做 $Y=H(x)$ , 而残差网络的一个残差块可以表示为 $H(x) = F(x)+x$，也就是$F(x) = H(x) - x$，在单位映射中， $y=x$ 便是观测值，而 $H(x)$ 是预测值，所以$F(x)$便对应着残差，因此叫做残差网络。

![](./img/res_block_0.png)

```python
class Residual(nn.Module): 
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)
```


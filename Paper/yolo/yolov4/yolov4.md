[TOC]

# Introduce

- [参考](https://www.cnblogs.com/pprp/p/12771430.html)

![structure](./img/yolov4-structure.png)

# 创新

## Mosaic

<img src="./img/Mosaic.jpg" style="zoom:150%;" />

将四张不同的图片嵌入到一张图中，其优点是：

- 混合四张具有不同语义信息的图片，可以让检测器检测超出常规语境的目标，增强模型的鲁棒性。
- 由于BN是从四张图片计算得到的，所以可以减少对大的mini-batch的依赖



## self-Adversarial Training

自对抗训练也是一种新的数据增强方法，可以一定程度上抵抗对抗攻击。其包括两个阶段，每个阶段进行一次前向传播和一次反向传播。

- 第一阶段，CNN通过反向传播改变图片信息，而不是改变网络权值。通过这种方式，CNN可以进行对抗性攻击，改变原始图像，造成图像上没有目标的假象。
- 第二阶段，对修改后的图像进行正常的目标检测。



## CMBN

![](./img/CMBN.jpg)

- BN是对当前mini-batch进行归一化。
- CBN是对当前以及当前往前数3个mini-batch的结果进行归一化。
- CmBN则逐渐累积

## SAM

![](./img/SAM.jpg)



## PAN

![](./img/PAN.jpg)



# The augmentation method used in paper

- [参考](https://mp.weixin.qq.com/s?__biz=MzUyMjE2MTE0Mw==&mid=2247489497&idx=2&sn=62a737bbec6c45dc46558b770d4c5306&chksm=f9d14941cea6c057330b805ace0f8b8d90b1878e562dfce681b77403ddfc739fc949c4c0abaf&scene=126&sessionid=1589551174&key=b053b5f9fd7f08e6870b241b0d81123decb0304ec6d626e61d33cc0ab67b09eb26a50da657495bca6d325ad00d33c41c6b5d9537fa75d902cea1d2d1058559d14a3bf853e21c0a5bbac99308e2a8e0d9&ascene=1&uin=MjUwOTM4MzI0MA%3D%3D&devicetype=Windows+10+x64&version=62090070&lang=zh_CN&exportkey=AeX%2BRyyFYZu3x6iaL9PvTq8%3D&pass_ticket=7%2BDndBTHQX4iS1onEy3XluyQaBW7CmGEgyO5danhG2Deo2FCa8PPQ7IhBoe40S4M)

## cutmix

- A method of mixing the cutout and mixup

![](./img/cutmix.png)

​	**mixup相当于是全图融合，cutout仅仅对图片进行增强，不改变label，而cutmix则是采用了cutout的局部融合思想，并且采用了mixup的混合label策略，看起来比较make sense。**

​    **cutmix和mixup的区别是，其混合位置是采用hard 0-1掩码，而不是soft操作,相当于新合成的两张图是来自两张图片的hard结合，而不是Mixup的线性组合。但是其label还是和mixup一样是线性组合**。
$$
\hat{x} = M \odot x_{A} + (1 - M) \odot x_{B} \\
\hat{y} = \lambda y_{A} + (1 - \lambda)y_{B}
$$
M是和原图大小一样的矩阵，只有0-1值，$\lambda$ 用于控制线性混合度，通过$\lambda$ 参数可以控制裁剪矩形的大小
$$
r_x \backsim Unif(0,W), r_w = W \sqrt{1 - \lambda} \\

r_y \backsim Unif(0,H), r_h = H \sqrt{1 - \lambda}
$$
![](./img/code-cutmix.png)

​	Mosaic增强是本文提出的，属于cutmix的扩展，**cutmix是两张图混合，而Mosaic增强是4张图混合**，好处非常明显是一张图相当于4张图，等价于batch增加了，可以显著减少训练需要的batch size大小。

![](./img/Mosaic.png)



## label smooth

- 核心就是对label进行soft操作，不要给0或者1的标签，而是有一个偏移，相当于在原label上增加噪声，让模型的预测值不要过度集中于概率较高的类别，把一些概率放在概率较低的类别。



## dropBlock

- [参考](https://www.pianshen.com/article/2769164511/)

​	dropBlock是在dropout上的推广。 dropout，训练阶段在每个mini-batch中，依概率P随机屏蔽掉一部分神经元，只训练保留下来的神经元对应的参数，屏蔽掉的神经元梯度为0，参数不参数与更新。而**测试阶段则又让所有神经元都参与计算**。

​	dropout操作流程：参数是丢弃率p
  1）在训练阶段，每个mini-batch中，按照伯努利概率分布(采样得到0或者1的向量，0表示丢弃)随机的丢弃一部分神经元（即神经元置零）。用一个mask向量与该层神经元对应元素相乘，mask向量维度与输入神经一致，元素为0或1。
  2）然后对**神经元rescale操作**，即每个神经元除以保留概率1-P,也即乘上1/(1-P)。
  3）反向传播只对保留下来的神经元对应参数进行更新。
  4）测试阶段，Dropout层不对神经元进行丢弃，保留所有神经元直接进行前向过程。

  为啥要rescale呢？是为了保证训练和测试分布尽量一致，或者输出能一致。可以试想，如果训练阶段随机丢弃，那么其实dropout输出的向量，有部分被屏蔽掉了，可以等下认为输出变了，如果dropout大量应用，那么其实可以等价为进行模拟遮挡的数据增强，如果增强过度，导致训练分布都改变了，那么测试时候肯定不好，**引入rescale可以有效的缓解，保证训练和测试时候，经过dropout后数据分布能量相似**。

```python
#!encoding=utf-8
import numpy as np
def dropout(x, drop_out_ratio,type="train"):
     # drop_out_ratio是概率值，必须在0~1之间
    if drop_out_ratio < 0. or drop_out_ratio>= 1: 
        raise Exception('Dropout level must be in interval [0, 1[.')
    if type=="train":
        scale = 1. / (1. - drop_out_ratio)
         # 即将生成一个0、1分布的向量，0表示这个神经元被屏蔽，不工作了，也就是dropout了
        mask_vec = np.random.binomial(n=1, p=1. - drop_out_ratio,size=x.shape) 
        print mask_vec
 
        x *= mask_vec                                                                            # 0、1与x相乘，我们就可以屏蔽某些神经元，让它们的值变为0
        x*=scale                                                                                     #再乘上scale系数
        print x
    return x
 
x=[100,29.5,1,2.0,3,4,23,12,34,45,667,76]
dratio=0.4
xnp=np.array(x)
 
print x
dropout(xnp,dratio,"train")
 
```

​	dropout方法多是作用在全连接层上，在卷积层应用dropout方法意义不大。文章认为是因为每个feature map的位置都有一个感受野范围，仅仅对单个像素位置进行dropout并不能降低feature map学习的特征范围，也就是说网络仍可以通过该位置的相邻位置元素去学习对应的语义信息，也就不会促使网络去学习更加鲁邦的特征。

 	 既然单独的对每个位置进行dropout并不能提高网络的泛化能力，那么很自然的，如果我们按照一块一块的去dropout，就自然可以促使网络去学习更加鲁邦的特征。思路很简单，就是**在feature map上去一块一块的找**，进行归零操作，类似于dropout，叫做dropblock。

![](./img/dropBlock.png)

 	绿色阴影区域是语义特征，b图是模拟dropout的做法，随机丢弃一些位置的特征，但是作者指出这做法没啥用，因为网络还是可以推断出来，(c)是dropBlock做法。

![](./img/algorithm-dropBlock.png)

​	dropblock有三个比较重要的参数，一个是block_size，用来控制进行归零的block大小；一个是γ，用来控制每个卷积结果中，到底有多少个channel要进行dropblock；最后一个是keep_prob，作用和dropout里的参数一样。
$$
\gamma = \frac{1 - keep\_prob}{block\_size ^ 2} 
\frac{feat\_size^2}{(feat\_size - block\_size + 1)}
$$
​	 M大小和输出特征图大小一致，非0即1，为了保证训练和测试能量一致，需要和dropout一样，进行rescale。

 	 上述是理论分析，在做实验时候发现，block_size控制为7*7效果最好，对于所有的feature map都一样，γ通过一个公式来控制，**keep_prob则是一个线性衰减过程，从最初的1到设定的阈值(具体实现是dropout率从0增加到指定值为止)，论文通过实验表明这种方法效果最好。如果固定prob效果好像不好。**实践中，并没有显式的设置![img](https://mmbiz.qpic.cn/mmbiz_png/iaTa8ut6HiawAn8osfpm51IkVQHh8abTX2cmicSYnodeb7daRser8ORmh7BibYosia2NppaHKiahAavTPrQLV847M9vA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)的值，而是根据keep_prob(具体实现是反的，是丢弃概率)来调整

  DropBlock in ResNet-50 DropBlock加在哪？**最佳的DropBlock配置是block_size=7，在group3和group4上都用**。将DropBlock用在skip connection比直接用在卷积层后要好，具体咋用，可以看代码。

```python

class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.

    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890

    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, 
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            # 最后计算的时候只要把那些为1的加起来，就是没有被drop的区域
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        # 比较巧妙的实现，用max pool来实现基于一点来得到全0区域
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
		# 经max_pool之后，宽和高会多出一行，去掉
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)
```


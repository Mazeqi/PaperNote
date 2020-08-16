[TOC]

# device

```python
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 model = Darknet(opt.model_def).to(device)
```



# tensor.permute

- 将tensor的维度换位。

```python
torch.Tensor.permute (Python method, in torch.Tensor)

>>> x = torch.randn(2, 3, 5) 
>>> x.size() 
torch.Size([2, 3, 5]) 
>>> x.permute(2, 0, 1).size() 
torch.Size([5, 2, 3])
```



# tensor.contiguous

- [参考](https://zhuanlan.zhihu.com/p/64376950)

```python
# 返回一个内存连续的有相同数据的tensor，如果原tensor内存连续，则返回原tensor；
# contiguous一般与transpose，permute，view搭配使用：使用transpose或permute进行维度变换后，调用contiguous，然后方可使用view对维度进行变形（如：tensor_var.contiguous().view() ），示例如下：

x = torch.Tensor(2,3)
y = x.permute(1,0)         # permute：二维tensor的维度变换，此处功能相当于转置transpose
y.view(-1)                 # 报错，view使用前需调用contiguous()函数
y = x.permute(1,0).contiguous()
y.view(-1)                 # OK

#1 transpose、permute等维度变换操作后，tensor在内存中不再是连续存储的，而view操作要求tensor的内存连续存储，所以需要contiguous来返回一个contiguous copy；

#2 维度变换后的变量是之前变量的浅拷贝，指向同一区域，即view操作会连带原来的变量一同变形，这是不合法的，所以也会报错；---- 这个解释有部分道理，也即contiguous返回了tensor的深拷贝contiguous copy数据；
```



# variable

- [参考](https://blog.csdn.net/u012370185/article/details/94391428)

Varibale包含三个属性：

- data：存储了Tensor，是本体的数据
- grad：保存了data的梯度，本事是个Variable而非Tensor，与data形状一致
- grad_fn：指向Function对象，用于反向传播的梯度计算之用

```python
# torch.autograd.Variable是Autograd的核心类，它封装了Tensor，并整合了反向传播的相关实现(tensor变成variable之后才能进行反向传播求梯度?用变量.backward()进行反向传播之后,var.grad中保存了var的梯度)

x = Variable(tensor, requires_grad = True)

#demo
import torch
from torch.autograd import Variable
 
x = Variable(torch.one(2,2), requires_grad = True)
print(x)#其实查询的是x.data,是个tensor
```



# save load

```python
# save
torch.save(model.state_dict(), PATH)

# load
model = MyModel(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
```



# torch.linspace

```python
torch.linspace(start, end, steps=100, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
'''
start：开始值
end：结束值
steps：分割的点数，默认是100
dtype：返回值（张量）的数据类型
'''

import torch
print(torch.linspace(3,10,5))
结果：tensor([ 3.0000, 4.7500, 6.5000, 8.2500, 10.0000])

type=torch.float
print(torch.linspace(-10,10,steps=6,dtype=type))
结果：tensor([-10., -6., -2., 2., 6., 10.])
```



# torch.clamp

```python
#将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。
# (n1, n2, 2)
intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0) 
torch.clamp(input, min, max, out=None) → Tensor
```



# torch.cuda.is_available

```python
#cuda是否可用；
torch.cuda.is_available()

# 返回gpu数量；
torch.cuda.device_count()

# 返回gpu名字，设备索引默认从0开始；
torch.cuda.get_device_name(0)

# 返回当前设备索引
torch.cuda.current_device()
```



# model.apply

```python
# demo1
def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        m.weight.data.fill_(1.0)
        print(m.weight)
        
net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
net.apply(init_weights)

#output
 Linear(in_features=2, out_features=2, bias=True)
Parameter containing:
    tensor([[ 1.,  1.],
            [ 1.,  1.]])
    Linear(in_features=2, out_features=2, bias=True)
Parameter containing:
    tensor([[ 1.,  1.],
            [ 1.,  1.]])
    Sequential(
        (0): Linear(in_features=2, out_features=2, bias=True)
        (1): Linear(in_features=2, out_features=2, bias=True)
    )
    Sequential(
        (0): Linear(in_features=2, out_features=2, bias=True)
        (1): Linear(in_features=2, out_features=2, bias=True)
    )
  
# yolov3
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
# model_def 为yolov3每个层的一些设置:
[convolutional]
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=leaky

model = Darknet(opt.model_def).to(device)
model.apply(weights_init_normal)
```



# torch.numel

```python
 torch.numel() 返回一个tensor变量内所有元素个数，可以理解为矩阵内元素的个数
```



# torch.squeeze &unsqueeze

```python
 torch.squeeze() 对于tensor变量进行维度压缩，去除维数为1的的维度。例如一矩阵维度为A*1*B*C*1*D，通过squeeze()返回向量的维度为A*B*C*D。squeeze(a)，表示将a的维数位1的维度删掉，squeeze(a,N)表示，如果第N维维数为1，则压缩去掉，否则a矩阵不变

torch.unsqueeze() 是squeeze()的反向操作，增加一个维度，该维度维数为1，可以指定添加的维度。例如unsqueeze(a,1)表示在1这个维度进行添加
```



# torch.stack

```python
torch.stack(sequence, dim=0, out=None)，做tensor的拼接。sequence表示Tensor列表，dim表示拼接的维度，注意这个函数和concatenate是不同的，torch的concatenate函数是torch.cat，是在已有的维度上拼接，而stack是建立一个新的维度，然后再在该纬度上进行拼接。
```



# torch.expand_as

```python
expand_as(a)这是tensor变量的一个内置方法，如果使用b.expand_as(a)就是将b进行扩充，扩充到a的维度，需要说明的是a的低维度需要比b大，例如b的shape是3*1，如果a的shape是3*2不会出错，但是是2*2就会报错了
```



# torch.view_as

```python
返回被视作与给定的tensor相同大小的原tensor。 等效于：
self.view(tensor.size())

# demo
a = torch.Tensor(2, 4)
b = a.view_as(torch.Tensor(4, 2))
```



# Dataset -> dataloader

## dataset  from torch.utils.data import Dataset

```python
class TensorDataset(Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
    """
        一般我们输入的tensors包含有 data和labels， 这里的getitem
         把每一条数据和label对应起来，并且形成tuple返回，然后我们
         就可以在dataloader之后这样做):
        for data in train_loader:
                images, labels = data
         函数中tensor[index]的index是由于data不止一个，这里留下
         index方便后面在dataloader中进行随机批量等操作。    
             
    """
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)
```



## DataLoader   from torch.utils.data DataLoader

- dataloaer:部分流程上有用的参数及其代码。

- 查看流程

- - 这部分代码首先要看shuffle，这里假设它是true，即跳转到处理shuffle，并生成sampler的那一行
  - BatchSamper那一行
  - __iter__，_SingleProcessDataLoaderIter的代码附在后面，

```python
class DataLoader(object):
    r"""
    Arguments:
       # dataset (Dataset): 一般从上面的tensordataset load进来
       # batch_size (int, optional): 每一次训练有多少个样本，默认是1 
       # shuffle (bool, optional): 是否打乱，默认False
       # sampler (Sampler, optional): 
       # 当sample不为none的时候，是不可以使用shuffle的，
         如果batch_sampler没有定义的话且batch_size有定义， 
          会根据sampler, batch_size, drop_last生成一个batch_sampler，
       # 当sample为none的时候，函数会根据shuffle是否为true生成一个
       乱序或者非乱序的sampler，后面的代码中观察出，sampler是数据的索引数组。
            
      	  batch_sampler (Sampler, optional): 
          # batch_sampler就是返回batch个sampler中的值，这些值是dataset的索引。
          所用当shuffle为true由于我们的限制，是不能直接传sampler进来的。因为当
          shuffle为true的时候，需要生成一个乱序的sampler。
      	  #num_workers (int, optional): 多线程，但是windows系统只能用0
 """
```



# torch.flip

```python
torch.flip(input, dims) → Tensor
沿给定轴反转nD张量的顺序，以暗淡表示.

>>> x = torch.arange(8).view(2, 2, 2)
>>> x
tensor([[[ 0,  1],
         [ 2,  3]],

        [[ 4,  5],
         [ 6,  7]]])
>>> torch.flip(x, [0, 1])
tensor([[[ 6,  7],
         [ 4,  5]],

        [[ 2,  3],
         [ 0,  1]]])
```

# torch.expend 

- 函数返回张量在某一个维度扩展之后的张量，就是将张量广播到新形状。函数对返回的张量不会分配新内存，即在原始张量上返回只读视图，返回的张量内存是不连续的。类似于numpy中的broadcast_to函数的作用。如果希望张量内存连续，可以调用contiguous函数。

```python
import torch
x = torch.tensor([1, 2, 3, 4])
xnew = x.expand(2, 4)
print(xnew)
# output
tensor([[1, 2, 3, 4],
        [1, 2, 3, 4]])
```

# torch.repeat

- torch.repeat用法类似np.tile，就是将原矩阵横向、纵向地复制。与torch.expand不同的是torch.repeat返回的张量在内存中是连续的。

```python
# 横向
import torch
x = torch.tensor([1, 2, 3])
xnew = x.repeat(1,3)
print(xnew)
# tensor([[1, 2, 3, 1, 2, 3, 1, 2, 3]])

# 纵向
import torch
x = torch.tensor([1, 2, 3])
xnew = x.repeat(3,1)
print(xnew)
# out
tensor([[1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]])

```



# torch.flatten()

- [参考](https://blog.csdn.net/jokerxsy/article/details/105968782)

- input: 输入，类型为Tensor。
- start_dim: 推平的起始维度。
- end_dim: 推平的结束维度。

```python
import torch

a = torch.ones(2,3,4,5)

b = torch.flatten(a,start_dim=0,end_dim=2)
# 从0维开始往后推，推到第2维。所以最后应该是:(2*3*4,5)
print(b.shape)

b = torch.flatten(a,end_dim=2)
# 默认为0
print(b.shape)

b = torch.flatten(a,start_dim=-1)
# 从最后一维往后退，不变
print(b.shape)

b = torch.flatten(a,end_dim=-1)
# 推到最后一维，展平
print(b.shape)

# output
torch.Size([24, 5])
torch.Size([24, 5])
torch.Size([2, 3, 4, 5])
torch.Size([120])
```



# F.interpolate

- [参考](https://blog.csdn.net/qq_41375609/article/details/103447744)

- input (Tensor) – 输入张量

- size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]) – 输出大小.

- scale_factor (float or Tuple[float]) – 指定输出为输入的多少倍数。如果输入为tuple，其也要制定为tuple类型

- mode (str) – 可使用的上采样算法，有’nearest’, ‘linear’, ‘bilinear’, ‘bicubic’ , ‘trilinear’和’area’. 默认使用’nearest’

```python
# 实现插值和上采样
def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
```



# pad

- [参考](https://zhuanlan.zhihu.com/p/95368411)

## pad = 'same'

- [参考](https://blog.csdn.net/baidu_36161077/article/details/81388141)

```python
# modify con2d function to use same padding
# code referd to @famssa in 'https://github.com/pytorch/pytorch/issues/3867'
# and tensorflow source code

import torch.utils.data
from torch.nn import functional as F

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple


class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# custom con2d, because pytorch don't have "padding='same'" option.
def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):

    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_needed = max(0, (out_rows - 1) * stride[0] + effective_filter_size_rows -
                  input_rows)
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)
```



# toch.nn

- [参考 ](https://www.cnblogs.com/wanghui-garcia/p/10791778.html) [loss参考](https://blog.csdn.net/dss_dssssd/article/details/84036913)

## conv init

- [参考1](https://www.zhihu.com/question/313869702) [参考2](https://blog.csdn.net/hyk_1996/article/details/82118797) [参考3](https://blog.csdn.net/dss_dssssd/article/details/83990511)

```python
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
                    
 # 1. 根据网络层的不同定义不同的初始化方式     
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式 
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
```

## Conv1d

$$
  L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor
$$





## Conv2d

- 二维卷积层, 输入的尺度是(N, Cin,H,W)，输出尺度（N,Cout,Hout,Wout）的计算方式： 
  $$
  out(N_i, C_{out_j}) = bias(C_{out_{j}}) + \sum^{C_{in}-1}_{k=0} wight(C_{out_{j}},k) * input(N_i, k)
  $$

- shape 输出的height和width的计算
  $$
  H_{out} = [\frac{H_{in} + 2 \times padding[0] - dilation[0] \times (kernel\_ size[0]-1)-1}
  {stride[0]}
  +1]
  
  \\
  W_{out} = [\frac{W_{in} + 2 \times padding[1] - dilation[1] \times (kernel\_ size[1]-1)-1}
  {stride[1]}
  +1]
  $$
  

```python
class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
'''
in_channels(int) – 输入信号的通道
out_channels(int) – 卷积产生的通道
kerner_size(int or tuple) - 卷积核的尺寸
stride(int or tuple, optional) - 卷积步长，默认为1
padding(int or tuple, optional) - 输入的每一条边补充0的层数，默认为0
dilation(int or tuple, optional) – 卷积核元素之间的间距,默认为1
groups(int, optional) – 从输入通道到输出通道的阻塞连接数。默认为1
bias(bool, optional) - 如果bias=True，添加可学习的偏置到输出中
'''
```



## ConvTranspose2d

`stride`: 控制相关系数的计算步长 
`dilation`: 用于控制内核点之间的距离
`groups`: 控制输入和输出之间的连接： 

- `group=1`，输出是所有的输入的卷积；
- `group=2`，此时相当于有并排的两个卷积层，每个卷积层计算输入通道的一半，并且产生的输出是输出通道的一半，随后将这两个输出连接起来。 

参数`kernel_size`，`stride`，`padding`，`dilation`数据类型：

- 可以是一个`int`类型的数据，此时卷积height和width值相同;
- 也可以是一个`tuple`数组（包含来两个`int`类型的数据），第一个`int`数据表示`height`的数值，第二个`int`类型的数据表示width的数值

$$
H_{out} = (H_{in}-1) \times stride[0] - 2 \times padding[0]+kernel\_ size[0] + output\_padding[0]
\\
W_{out} = (W_{in}-1) \times stride[1] - 2 \times padding[1]+kernel\_ size[1] + output\_padding[1]
$$

- in_channels(int) – 输入信号的通道数
- out_channels(int) – 卷积产生的通道数
- kerner_size(int or tuple) - 卷积核的大小
- stride(int or tuple,optional) - 卷积步长
- padding(int or tuple, optional) - 输入的每一条边补充padding= kernel - 1 - padding,即(kernel_size - 1)/20的层数，所以补充完高宽都增加(kernel_size - 1)
- output_padding(int or tuple, optional) - 在输出的每一个维度的一边补充0的层数，所以补充完高宽都增加padding，而不是2*padding，因为只补一边
- dilation(int or tuple, optional) – 卷积核元素之间的间距
- groups(int, optional) – 从输入通道到输出通道的阻塞连接数
- bias(bool, optional) - 如果bias=True，添加偏置

```python
# 对由多个输入平面组成的输入图像应用二维转置卷积操作。
class torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
```



## BatchNorm2d

```python
# 对小批量(mini-batch)3d数据组成的4d输入进行批标准化(Batch Normalization)操作
# xnew = (1-momentum) * x + momentum * xt
'''
num_features： C来自期待的输入大小(N,C,H,W)
eps： 即上面式子中分母的ε ，为保证数值稳定性（分母不能趋近或取0）,给分母加上的值。默认为1e-5。
momentum： 动态均值和动态方差所使用的动量。默认为0.1。
affine： 一个布尔值，当设为true，给该层添加可学习的仿射变换参数，即γ与β。
track_running_stats：一个布尔值，当设置为True时，该模块跟踪运行的平均值和方差，当设置为False时，该模块不跟踪此类统计数据，并且始终在train和eval模式中使用批处理统计数据。默认值:True
'''
class torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True)
```



## MaxPool2d

$$
\begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
\end{aligned}
\\
 H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                    \times (\text{kernel_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor
\\
 W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                    \times (\text{kernel_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor
$$



```python
class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
'''
kernel_size(int or tuple) - max pooling的窗口大小
stride(int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size
padding(int or tuple, optional) - 输入的每一条边补充0的层数
dilation(int or tuple, optional) – 一个控制窗口中元素步幅的参数
return_indices - 如果等于True，会返回输出最大值的序号，对于上采样操作torch.nn.MaxUnpool2d会有帮助
ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整ceil，代替默认的向下取整floor的操作
'''
```



## AdaptiveAvgPool2d

```python
class torch.nn.AdaptiveAvgPool2d(output_size)
# output_size: 输出信号的尺寸,可以用(H,W)表示H*W的输出，也可以使用单个数字H表示H*H大小的输出
# 对输入信号，提供2维的自适应平均池化操作
# ---------------------------------------
input = torch.randn(1,64,8,9)
m = nn.AdaptiveAvgPool2d((5,7))
output = m(input)
output.shape

# output
torch.Size([1, 64, 5, 7])

```



## ReLU

```python
class torch.nn.ReLU(inplace=False)
# 对输入运用修正线性单元函数：
# {ReLU}(x)= max(0, x)
# inplace-选择是否进行原位运算，即x= x+1
```



## LeakyReLU

$$
\begin{eqnarray}

& LeakyReLU(x) & = max(0,x) +negative\_slope * min(0,x) & \\

& or   		   &  \\

& LeakyReLU(x) & = 
\begin{cases}
		x, 						  &&&&  if \ \ x\geq 0 \\
		negative\_slope \times x, &&&&  otherwise
\end{cases}
\\

\end{eqnarray}
$$



```python
class torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
# 是ReLU的变形，Leaky  ReLU是给所有负值赋予一个非零斜率
# negative_slope：控制负斜率的角度，默认等于0.01
# inplace-选择是否进行原位运算，即x= x+1，默认为False
```



## Sigmoid

$$
Sigmoid(X) = \frac{1}{1 + \exp(-x)}
$$



```python
class torch.nn.Sigmoid
# 输出值的范围为[0,1]

# demo
m = nn.Sigmoid()
input = torch.randn(2)
output = m(input)
input, output

(tensor([-0.8425,  0.7383]), tensor([0.3010, 0.6766]))
```



## Tanh

$$
Tanh(x) = tanh(x) = \frac{
e^x - e^{-x}
}{
e^x + e^{-x}
}
$$

```python
class torch.nn.Tanh

# demo
m = nn.Tanh()
input = torch.randn(2)
output = m(input)
input, output
(tensor([-0.6246,  0.1523]), tensor([-0.5543,  0.1512]))
```



## BCELoss

- **weight** (*Tensor**,**可选*) – a manual rescaling weight given to the loss of each batch element. If given, has to be a Tensor of size “nbatch”.每批元素损失的手工重标权重。如果给定，则必须是一个大小为“nbatch”的张量。
- **size_average** (*bool**, 可选*) – `弃用(见reduction参数)。默认情况下，设置为True，即对批处理中的每个损失元素进行平均。注意，对于某些损失，每个样本有多个元素。如果字段size_average设置为False，则对每个小批的损失求和。当reduce为False时，该参数被忽略。默认值:True`
- **reduce** (*bool**,**可选*) – `弃用(见reduction参数)。默认情况下，设置为True，即根据size_average参数的值决定对每个小批的观察值是进行平均或求和。如果reduce为False，则返回每个批处理元素的损失，不进行平均和求和操作，即忽略size_average参数。默认值:True`
- **reduction** (*string**,**可选*) – 指定要应用于输出的`reduction`操作:' none ' | 'mean' | ' sum '。“none”:表示不进行任何`reduction`，“mean”:输出的和除以输出中的元素数，即求平均值，“sum”:输出求和。注意:size_average和reduce正在被弃用，与此同时，指定这两个arg中的任何一个都将覆盖reduction参数。默认值:“mean”

```python
class torch.nn.BCELoss(weight=None, size_average=True, reduce=None, reduction='mean')
# 计算 target 与 output 之间的二进制交叉熵。

m = nn.Sigmoid()
loss = nn.BCELoss()
input = torch.randn(3,requires_grad=True)
target = torch.empty(3).random_(2)
output = loss(m(input), target)
output.backward()

# output
(tensor([-0.8728,  0.3632, -0.0547], requires_grad=True),
 tensor([1., 0., 0.]),
 tensor(0.9264, grad_fn=<BinaryCrossEntropyBackward>))
```

## NLLLoss

- [参考](https://blog.csdn.net/weixin_40476348/article/details/94562240)

常用于多分类任务，NLLLoss 函数输入 input 之前，需要对 input 进行 log_softmax 处理，即将 input 转换成概率分布的形式，并且取对数，底数为e

```python
class torch.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, 
					   reduce=None, reduction='mean')
```

计算公式：loss(input, class) = -input[class]
公式理解：input = [-0.1187, 0.2110, 0.7463]，target = [1]，那么 loss = -0.2110
官方文档中介绍称： nn.NLLLoss输入是一个对数概率向量和一个目标标签，它与nn.CrossEntropyLoss的关系可以描述为：softmax(x)+log(x)+nn.NLLLoss====>nn.CrossEntropyLoss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(2019)

output = torch.randn(1, 3)  # 网络输出
target = torch.ones(1, dtype=torch.long).random_(3)  # 真实标签
print(output)
print(target)

# 直接调用
loss = F.nll_loss(output, target)
print(loss)

# 实例化类
criterion = nn.NLLLoss()
loss = criterion(output, target)
print(loss)

"""
tensor([[-0.1187,  0.2110,  0.7463]])
tensor([1])
tensor(-0.2110)
tensor(-0.2110)
"""

```

如果 input 维度为 M x N，那么 loss 默认取 M 个 loss 的平均值，reduction='none' 表示显示全部 loss

```
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(2019)

output = torch.randn(2, 3)  # 网路输出
target = torch.ones(2, dtype=torch.long).random_(3)  # 真实标签
print(output)
print(target)

# 直接调用
loss = F.nll_loss(output, target)
print(loss)

# 实例化类
criterion = nn.NLLLoss(reduction='none')
loss = criterion(output, target)
print(loss)

"""
tensor([[-0.1187,  0.2110,  0.7463],
        [-0.6136, -0.1186,  1.5565]])
tensor([2, 0])
tensor(-0.0664)
tensor([-0.7463,  0.6136])
"""

```


## CrossEntropyLoss

- [参考](https://blog.csdn.net/Jeremy_lf/article/details/102725285)

log_softmax是指在softmax函数的基础上，再进行一次log运算，此时结果有正有负，log函数的值域是负无穷到正无穷，当x在0—1之间的时候，log(x)值在负无穷到0之间。

```python
CrossEntropyLoss()=log_softmax() + NLLLoss() 

loss=torch.nn.NLLLoss()
target=torch.tensor([0,1,2])
loss(input,target)
Out[26]: tensor(-0.1399)
loss =torch.nn.CrossEntropyLoss()
input = torch.tensor([[ 1.1879,  1.0780,  0.5312],
        [-0.3499, -1.9253, -1.5725],
        [-0.6578, -0.0987,  1.1570]])
target = torch.tensor([0,1,2])
loss(input,target)
Out[30]: tensor(0.1365)

# -----------------------------------------------------------------------------------
# 如果传给target是one_hot编码格式呢？
# 将target one_hot的编码格式转换为每个样本的类别，再传给CrossEntropyLoss
import torch

from torch import nn
from torch.nn import functional as F
# 编码one_hot
def one_hot(y):
    '''
    y: (N)的一维tensor，值为每个样本的类别
    out: 
        y_onehot: 转换为one_hot 编码格式 
    '''
    y = y.view(-1, 1)
    y_onehot = torch.FloatTensor(3, 5)
    
    # In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    return y_onehot


def cross_entropy_one_hot(input_, target):
	# 解码 
    _, labels = target.max(dim=1)
    # 调用cross_entropy
    return F.cross_entropy(input_, labels)

# 调用计算loss： loss_1 = cross_entropy_one_hot(x, one_hot(y))


```





## MSELoss

- 计算输入x和标签y，n个元素平均平方误差（mean square error），x和y具有相同的Size

$$
l(x,y) = L = \{l_1,...,l_N\}^T, l_n = (x_n -y_n)^2
$$

- 如果reduction != ‘none’:

$$
l(x,y) = 
\begin{cases}
	mean(L), 						  &&&&  if \ \ reduction = 'elementwise\_mean',
	\\
	sum(L) ,                          &&&&  if \ \ reduction = 'sum'
\end{cases}
$$



```python
class torch.nn.MSELoss(size_average=None, reduce=None, reduction='elementwise_mean')

#demo
import torch
from torch import nn
criterion_none = nn.MSELoss( reduction='none')
criterion_elementwise_mean = nn.MSELoss(reduction='elementwise_mean')
criterion_sum = nn.MSELoss(reduction='sum')

x = torch.randn(3, 2, requires_grad=True)
y = torch.randn(3, 2)

loss_none = criterion_none(x, y)

loss_elementwise_mean = criterion_elementwise_mean(x, y)

loss_sum = criterion_sum(x, y )

print('reduction={}:   {}'.format('none', loss_none.detach().numpy()))
print('reduction={}:   {}'.format('elementwise_mean', loss_elementwise_mean.item()))
print('reduction={}:   {}'.format('sum', loss_sum.item()))

#output
reduction=none:
[[0.02320575 0.30483633]
[0.04768182 0.4319028 ]
[3.11864 7.9872203 ]]
reduction=elementwise_mean: 1.9855811595916748 # 1.9 * 6 = 11.4
reduction=sum: 11.913487434387207
```



## DataParallel

- [参考](https://blog.csdn.net/qq_19598705/article/details/80396325)

- 多GPU使用

```python
DEVICE_INFO=dict(
    gpu_num=torch.cuda.device_count(),
    device_ids = range(0, torch.cuda.device_count(), 1),
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu",
    index_cuda=0, )

device   = device_info['device']
PrallelModel = torch.nn.DataParallel(model, device_ids = device_info['device_ids'])
PrallelModel.to(device)

```

- 单gpu

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```



## Upsample

- [参考](https://www.cnblogs.com/wanghui-garcia/p/11399053.html)

$$
D_{out} = \left\lfloor D_{in} \times \text{scale_factor} \right\rfloor
\\
 H_{out} = \left\lfloor H_{in} \times \text{scale_factor} \right\rfloor
 \\
         W_{out} = \left\lfloor W_{in} \times \text{scale_factor} \right\rfloor
$$



-  size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional)
- scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional)
- mode (str, optional): the upsampling algorithm: one of ``'nearest'``, ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``. Default: ``'nearest'``
- align_corners (bool, optional): if ``True``, the corner pixels of the input and output tensors are aligned, and thus preserving the values at those pixels. This only has effect when :attr:`mode` is  ``'linear'``, ``'bilinear'``, or ``'trilinear'``. Default: ``False``

```python
torch.nn.Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None)


>>> input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
>>> input
tensor([[[[ 1.,  2.],
          [ 3.,  4.]]]])

>>> m = nn.Upsample(scale_factor=2, mode='nearest')
>>> m(input)
tensor([[[[ 1.,  1.,  2.,  2.],
          [ 1.,  1.,  2.,  2.],
          [ 3.,  3.,  4.,  4.],
          [ 3.,  3.,  4.,  4.]]]])
```





# torch.optim

## optimizer.step

```python
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```



## SGD

```python
class torch.optim.SGD(params, lr=<object object>, momentum=0,
dempening=0, weight_decay=0, nesterov=False)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer.zero_grad() # clears the gradients of all optimized **Variable** s.
loss_fn(model(input), target).backward()
optimizer.step() # performs a single optimization step.
```



## Adam

```python
class torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999),
eps=1e-08, weight_decay=0)
```



## lr_scheduler.ReduceLROnPlateau

- [参考](https://blog.csdn.net/weixin_40100431/article/details/84311430)

- optimer指的是网络的优化器
- mode (str) ，可选择‘min’或者‘max’，min表示当监控量停止下降的时候，学习率将减小，max表示当监控量停止上升的时候，学习率将减小。默认值为‘min’
- factor 学习率每次降低多少，new_lr = old_lr * factor
- patience=10，容忍网路的性能不提升的次数，高于这个次数就降低学习率
- verbose（bool） - 如果为True，则为每次更新向stdout输出一条消息。 默认值：False
- threshold（float） - 测量新最佳值的阈值，仅关注重大变化。 默认值：1e-4
- cooldown： 减少lr后恢复正常操作之前要等待的时期数。 默认值：0。
- min_lr,学习率的下限
- eps ，适用于lr的最小衰减。 如果新旧lr之间的差异小于eps，则忽略更新。 默认值：1e-8。

```python
class torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
 verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=4, verbose=True)
.....
scheduler.step(train_loss)
# scheduler.step(val_loss)

```


[TOC]

# device

```python
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 model = Darknet(opt.model_def).to(device)
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

## dataset

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



## DataLoader

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




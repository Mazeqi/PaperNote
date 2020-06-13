[TOC]

# tensorflow

## image

- tensorflow的输入图片的形状是 [batch_size, image_size, image_size, 3]

```python
images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images')
```



## tf.tile

```python
#在batch_size的纬度乘了batch_size,原来这个纬度的通道是1
        self.offset = tf.tile(self.offset, (self.batch_size, 1, 1, 1))
```



## tf.placeholder

```python
#占位符
self.labels = tf.placeholder(tf.float32, [None, self.cell_size, self.cell_size, self.box_per_cell, self.num_class + 5], name = 'labels')
```



## tf.nn.max_pool

```python
#ksize是kernel的大小，一般是[1,x,x,1]第一个和最后一个纬度一般不做池化
# padding='SAME'使得加入填充后，输出与输入的形状一致
pool = tf.nn.max_pool(inputs, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = name)

```



## tf.nn.conv2d

```python
# filter = [filter_height, filter_width, in_channels, out_channels] [3,3,3,32] filter_height 为kernel的高
tf.nn.conv2d(inputs, filter, strides=[1, 1, 1, 1], padding='SAME', name=name)
```



## tf.layer.conv2d

```python
# filters_num 是通道数 kernel_size是卷积核大小
out=tf.layers.conv2d(x,filters=filters_num,kernel_size=filters_size,strides=stride,padding='VALID',activation=None,use_bias=use_bias,name=name)
```



## tf.nn.batch_normalization

```python
# `output = (x - mean) / (sqrt(var) + epsilon) * gamma + beta`
# y=scale∗(x−mean)/var+offset
# 得到通道数，初始图片是3，卷积后为各种长度
depth = shape[3]

# 缩放，默认是1,归一化
scale = tf.Variable(tf.ones([depth, ], dtype='float32'), name='scale')

# 偏移量，beta
shift = tf.Variable(tf.zeros([depth, ], dtype='float32'), name='shift')

# 均值
mean = tf.Variable(tf.ones([depth, ], dtype='float32'), name='rolling_mean')

# 方差
variance = tf.Variable(tf.ones([depth, ], dtype='float32'), name='rolling_variance')

# 1e-05防止除数为0
conv_bn = tf.nn.batch_normalization(conv, mean, variance, shift, scale, 1e-05)
```



## tf.layer.batch_normalize

```python
out=tf.layers.batch_normalization(out,axis=-1,momentum=0.9,training=False,name=name+'_bn')
```



## tf.shape tensor.getshape

```python
# 返回元组，不能放到sess.run()里面
tensor.get_shape()

# 返回一个tensor，想要知道多少，必须放到sess.run()
tf.shape()
```



## tf.stack tf.concat

- [参考](https://zhuanlan.zhihu.com/p/37637446)

```python
# tf.concat是沿某一维度拼接shape相同的张量，拼接生成的新张量维度不会增加。而tf.stack是在新的维度上拼接，拼接后维度加1
ab1 = tf.concat([a,b],axis=0)
ab2 = tf.stack([a,b], axis=0)
```



## tf.clip_by_value

```python
#输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。小于min的让它等于min，大于max的元素的值等于max。
tf.clip_by_value(A, min, max)：
```





# torch

## torch.clamp

```python
#将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。
# (n1, n2, 2)
intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0) 
torch.clamp(input, min, max, out=None) → Tensor
```


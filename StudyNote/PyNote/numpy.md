[TOC]



# np.arange

```python
#生成几个相同的数组并合成一个shape = [20,5]
numpy.arange(start, stop, step, dtype)
[np.arange(5)] * 4 * 5
```



# 双冒号

```python
seq[start:end:step]
```



# np.dtype np.size...

```python
import numpy as np  
  
a1 = np.array([1,2,3,4],dtype=np.complex128)  
print(a1)  
print("数据类型",type(a1))           #打印数组数据类型  
print("数组元素数据类型：",a1.dtype) #打印数组元素数据类型  
print("数组元素总数：",a1.size)      #打印数组尺寸，即数组元素总数  
print("数组形状：",a1.shape)         #打印数组形状  
print("数组的维度数目",a1.ndim)      #打印数组的维度数目  
```



# np.tile

- [参考](https://www.jianshu.com/p/4b74a367833c)

```python
tile(A, reps)
    Construct an array by repeating A the number of times given by reps.
    
    If `reps` has length ``d``, the result will have dimension of
    ``max(d, A.ndim)``.
    
    If ``A.ndim < d``, `A` is promoted to be d-dimensional by prepending new
    axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication,
    or shape (1, 1, 3) for 3-D replication. If this is not the desired
    behavior, promote `A` to d-dimensions manually before calling this
    function.
    
    If ``A.ndim > d``, `reps` is promoted to `A`.ndim by pre-pending 1's to it.
    Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as
    (1, 1, 2, 2).
    
    Note : Although tile may be used for broadcasting, it is strongly
    recommended to use numpy's broadcasting operations and functions.
    
    Parameters
    ----------
    A : array_like
        The input array.
    reps : array_like
        The number of repetitions of `A` along each axis.
    
    Returns
    -------
    c : ndarray
        The tiled output array.
    
    See Also
    --------
    repeat : Repeat elements of an array.
    broadcast_to : Broadcast an array to a new shape
    
    Examples
    --------
    >>> a = np.array([0, 1, 2])
    >>> np.tile(a, 2)
    array([0, 1, 2, 0, 1, 2])
    >>> np.tile(a, (2, 2))
    array([[0, 1, 2, 0, 1, 2],
           [0, 1, 2, 0, 1, 2]])
    >>> np.tile(a, (2, 1, 2))
    array([[[0, 1, 2, 0, 1, 2]],
           [[0, 1, 2, 0, 1, 2]]])
    
    >>> b = np.array([[1, 2], [3, 4]])
    >>> np.tile(b, 2)
    array([[1, 2, 1, 2],
           [3, 4, 3, 4]])
    >>> np.tile(b, (2, 1))
    array([[1, 2],
           [3, 4],
           [1, 2],
           [3, 4]])
    
    >>> c = np.array([1,2,3,4])
    >>> np.tile(c,(4,1))
    array([[1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4]])
```



# np.nonzero

- [参考](https://blog.csdn.net/zhihaoma/article/details/51235016)

```python
# nonzero(a)返回数组a中值不为零的元素的下标，它的返回值是一个长度为a.ndim(数组a的轴数)的元组，元组的每个元素都是一个整数数组，其值为非零元素的下标在对应轴上的值。例如对于一维布尔数组b1，nonzero(b1)所得到的是一个长度为1的元组，它表示b1[0]和b1[2]的值不为0(False)。

>>> b1=np.array([True, False, True, False])
>>> np.nonzero(b1)
(array([0, 2], dtype=int64),)


# 对于二维数组b2，nonzero(b2)所得到的是一个长度为2的元组。它的第0个元素是数组a中值不为0的元素的第0轴的下标，第1个元素则是第1轴的下标，因此从下面的结果可知b2[0,0]、b[0,2]和b2[1,0]的值不为0：
>>> b2 = np.array([[True, False, True], [True, False, False]])
>>> np.nonzero(b2)
(array([0, 0, 1], dtype=int64), array([0, 2, 0], dtype=int64))

# demo
dataSet=array(
	[[1,0,0,0],
	[0,1,0,0],
	[0,1,0,0],
	[0,0,0,1]])
a=dataSet[:,1]>0.5
print(a)
print('--------------')
print(nonzero(a))
print('--------------')
print(nonzero(a)[0])
print('--------------')
print(dataSet[nonzero(a)[0],:]

# 结果
[False  True  True False]
--------------
(array([1, 2], dtype=int64),)
--------------
[1 2]
--------------
[[0 1 0 0]
 [0 1 0 0]]
      

# yolov2
# 达到阈值
filter_probs = np.array(probs >= self.threshold, dtype = 'bool')
# 返回达到阈值的box的下标
filter_index = np.nonzero(filter_probs)
# 筛选box
box_filter = boxes[filter_index[0], filter_index[1], filter_index[2]]
```



# numpy.axis

- [参考1](https://blog.csdn.net/weixin_38145317/article/details/79650188)
- [参考2](https://blog.csdn.net/sky_kkk/article/details/79725646)

```python
# 首先对numpy中axis取值进行说明：一维数组时axis=0，二维数组时axis=0，1，维数越高，则axis可取的值越大，数组n维时，axis=0，1，…，n。为了方便下面的理解，我们这样看待：在numpy中数组都有着[]标记，则axis=0对应着最外层的[]，axis=1对应第二外层的[]，以此类推，axis=n对应第n外层的[]。

#demo 1
# 有两层[]，最外层[]里的最大单位块分别为[1,2]，[3,4]，对这两个单位块做块与块之间的运算，[1,2]+[3,4] = [4, 6]；
# 做完加法后本应是[[4, 6]]，但是移除最外层[]后，原来的两层[]变成一层[],所以返回结果为 [4, 6]。
a= np.array([
    [1,2],
    [3,4]
])  
a.sum(axis = 0)
>>>array([4, 6])

#demo2
a = np.array([#这个方括号不算
    	
     #第二这个方括号是shape[0]共有三个
              [
                #这里三行是shape[1]  
                  [1, 5, 5, 2],
                  
                #每个方括号的横向是shape[2]
                  [9, -6, 2, 8],
                  [-3, 7, -9, 1]
              ],
 
              [
                  [-1, 7, -5, 2],
                  [9, 6, 2, 8],
                  [3, 7, 9, 1]
              ],
            
             [
                  [21, 6, -5, 2],
                  [9, 36, 2, 8],
                  [3, 7, 79, 1]
              ]
])

>>> a.shape
(3,3,4)

'''
 在axis = 0的方向上进行比较，如
 [1, 5, 5, 2],
 [-1, 7, -5, 2],
 [21, 6, -5, 2]
 拿出每一个数组的第一行出来比较，并在行的位置上进行比较，最大的分别是21， 7， 5， 2
 所以为[2,1,0,0]
'''
>>> np.argmax(a, axis=0)
array([[2, 1, 0, 0],
       [0, 2, 0, 0],
       [1, 0, 2, 0]], dtype=int64)

'''
在axis = 1的方向上进行比较，如比较第一个
  [ 1  5  5  2]
  [ 9 -6  2  8]
  [-3  7 -9  1]
  从axis=1的方向出发，第一个1,9,-3中，9最大，所以是1，依此类推则为[1,2,0,1]
'''
>>> np.argmax(a, axis=1)
array([[1, 2, 0, 1],
       [1, 0, 2, 1],
       [0, 1, 2, 1]], dtype=int64)

```



# c = [True, False]  a[c]

```python
c = np.array([
    [True,False],
    [True,False]
])

a = np.array([
    [1,2],
    [3,4]
])

print(a[c])
>>> [1,4] # 只输出True的值
```



# np.argsort

- [参考](https://www.jianshu.com/p/64c607d49528)

```python
argsort(a, axis=-1, kind='quicksort', order=None)

    >>> x = np.array([3, 1, 2])
    >>> y = np.argsort(x)
    # 在x中，1最小，所以他要被放入到第一位，所以y[0]的值是1的索引1,
    # 在x中，3最大，所以他要被放入到最后, 所以y[2]的值是3的索引0
        array([1, 2, 0])
    
Two-dimensional array:
    
    >>> x = np.array([[0, 3], [2, 2]])
    >>> x
    array([[0, 3],
           [2, 2]])

    >>> np.argsort(x, axis=0)
    array([[0, 1],
           [1, 0]])

    >>> np.argsort(x, axis=1)
    array([[0, 1],
           [0, 1]])
```



# np.fromfile

- [参考](https://www.jianshu.com/p/e9f6b15318be)



# np.random.binomial(n,p,size = None)

- 这是二项分布
- n：int型或者一个int型的数组，大于等于0，接受浮点数但是会被变成整数来使用。
- p：float或者一组float的数组，大于等于0且小于等于1.
- size：可选项，int或者int的元祖，表示的输出的大小，如果提供了size，例如(m,n,k)，那么会返回m*n*k个样本。如果size=None，也就是默认没有的情况，当n和p都是一个数字的时候只会返回一个值，否则返回的是np.broadcast(n,p).size个样本.
- return :一个数字或者一组数字,每个样本返回的是n次试验中事件A发生的次数。



# np.random.randint

```python
numpy.random.randint(low,high=None,size=None,dtype)
# 生成在半开半闭区间[low,high)上离散均匀分布的整数值;若high=None，则取值区间变为[0,low)
```



# np.random.random_integers()

```python
numpy.random.random_integers(low,high=None,size=None)
# 生成闭区间[low,high]上离散均匀分布的整数值;若high=None，则取值区间变为[1,low]
```



# np.random.randn()

```python
numpy.random.rand(d0,d1,…dn)
# 以给定的形状创建一个数组，数组元素来符合标准正态分布N(0,1)
```



# np.random.rand()

```python
numpy.random.rand(d0,d1,…dn)
#以给定的形状创建一个数组，并在数组中加入在[0,1]之间均匀分布的随机样本。
```



# numpy.random.seed() & np.random.RandomState()

```python
np.random.seed(10)
#这两个在数据处理中比较常用的函数，两者实现的作用是一样的，都是使每次随机生成数一样
```



# np.random.choice

```python
numpy.random.choice(a,size=None,replace=True,p=None)
# 若a为数组，则从a中选取元素；若a为单个int类型数，则选取range(a)中的数
# replace是bool类型，为True，则选取的元素会出现重复；反之不会出现重复
# p为数组，里面存放自己输入的选到每个数的可能性，即概率
```



# numpy.random_sanmple()

- [参考](https://blog.csdn.net/m0_38061927/article/details/75335069)

```python
numpy.random.random_sample(size=None)
# 以给定形状返回[0,1)之间的随机浮点数
```



# np.random.normal

- [参考](https://blog.csdn.net/lanchunhui/article/details/50163669)

- 高斯分布密度函数

$$
f(x) = \frac{1}{\sqrt{2\pi} \rho} \exp(- \frac{(x - u)^2}{2 \rho^2})​
$$



```python
numpy.random.normal(loc=0.0, scale=1.0, size=None)
'''
loc：float
    此概率分布的均值（对应着整个分布的中心centre）
scale：float
    此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高）
size：int or tuple of ints
    输出的shape，默认为None，只输出一个值
'''
```



# np.random.uniform

- [参考](https://blog.csdn.net/u013920434/article/details/52507173)

```python
numpy.random.uniform(low,high,size)
# 从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
# low: 采样下界，float类型，默认值为0；
# high: 采样上界，float类型，默认值为1；
# size: 输出样本数目，为int或元组(tuple)类型，例如，size=(m,n,k), 则输出m*n*k个样本，缺省时输出1个值。
```



# np to add noise

- [参考](https://blog.csdn.net/weixin_44191286/article/details/86437924)

```python
# 1.产生噪音那使用开根号是因为开根号后的数乘上所有的噪音信号，这样在计算噪音强度的时候可以直接提出来。
# 2.因为原本的噪音本身是有强度的所以需要除一下，而添加高斯白噪音的时候就不需要添加，是因为高斯白噪声本身的强度和其方差是一样的，是1

# 信噪比： snr =10 log_{10} (P_{singal} / p_{noise})
#原始噪音(d)的信号强度：P_d
#所需要的噪音强度：P_noise=P_signal / 10**(SNR / 10)
#所需要产生的噪音：noise=np.sqrt(P_noise / P_d) * d
# 产生的噪音强度为(SNR)的含噪信号：NoiseSignal = x + noise
def Add_noise(x, d, SNR):
     P_signal=np.sum(abs(x)**2)
     P_d=np.sum(abs(d)**2)
     P_noise=P_signal/10**(SNR/10)
     noise=np.sqrt(P_noise/P_d)*d
     return noise_signal=x+noise



# SNR_db=kwargs.get("noise_SNR_db",[5,15])
def jitter(x, snr_db):
    """
    根据信噪比添加噪声
    :param x:
    :param snr_db: [min, max]
    :return:
    """
    # 随机选择信噪比
    assert isinstance(snr_db, list)
    snr_db_low = snr_db[0]
    snr_db_up = snr_db[1]
    snr_db = np.random.randint(snr_db_low, snr_db_up, (1,))[0]
    # 信噪比, 要先除以10然后才能得出
    snr = 10 ** (snr_db / 10)
    
    Xp = np.sum(x ** 2, axis=0, keepdims=True) / x.shape[0]  # 计算信号功率
    Np = Xp / snr  # 计算噪声功率
    n = np.random.normal(size=x.shape, scale=np.sqrt(Np), loc=0.0)  # 计算噪声
    xn = x + n
    return xn

# 高斯白噪声
def wgn(x, snr):
	 P_signal = np.sum(abs(x)**2)/len(x)
 　　P_noise = P_signal/10**(snr/10.0)
 　　return np.random.randn(len(x)) * np.sqrt(P_noise)

```



# np to normalize

- [参考](https://www.jianshu.com/p/0d8bb02f98fb)

- **线性归一化 **[0,1]之间

$$
x^` = \frac{x - min(x)}{max(x) - min(x)}
$$

```python
def Normalization(x):
    return [(float(i)-min(x))/float(max(x)-min( x)) for i in x]
    
# 或者调用sklearn包的方法
from sklearn import preprocessing   
import numpy as np  
X = np.array([[ 1., -1.,  2.],  
              [ 2.,  0.,  0.],  
              [ 0.,  1., -1.]])  
min_max_scaler = preprocessing.MinMaxScaler()  
X_minMax = min_max_scaler.fit_transform(X)  

```

- **线性归一化 **[-1,1]之间

$$
x^* = \frac{x - x_{mean}}{x_{max} - x_{min}}
$$

```python
def Normalization2(x):
    return [(float(i)-np.mean(x))/(max(x)-min(x)) for i in x]
```

- **标准差化**

  也称为z-score标准化。这种方法根据原始数据的均值（mean）和标准差（standard deviation）进行数据的标准化。经过处理的数据符合标准正态分布，即均值为0，标准差为1，其转化函数为：
  $$
  x^* = \frac{x - u}{\rho}
  $$
  其中μ为所有样本数据的均值，σ为所有样本数据的标准差。

> 在分类、聚类算法中，需要使用距离来度量相似性的时候、或者使用PCA技术进行降维的时候，Z-score标准化表现更好。

```python
from sklearn import preprocessing   
import numpy as np  
X = np.array([[ 1., -1.,  2.],  
              [ 2.,  0.,  0.],  
              [ 0.,  1., -1.]])  
# calculate mean  
X_mean = X.mean(axis=0)  
# calculate variance   
X_std = X.std(axis=0)  
# standardize X  
X1 = (X-X_mean)/X_std  # 自己计算
# use function preprocessing.scale to standardize X  
X_scale = preprocessing.scale(X)  # 调用sklearn包的方法
# 最终X1与X_scale等价
```

- **非线性归一化**

  经常用在数据分化比较大的场景，有些数值很大，有些很小。通过一些数学函数，将原始值进行映射。该方法包括 log、指数，正切等。需要根据数据分布的情况，决定非线性函数的曲线，比如log(V, 2)还是log(V, 10)等。



# np.concatenate

- 传入的参数必须是**一个多个数组的元组或者列表**
- 另外需要指定拼接的方向，默认是 axis = 0，也就是说对0轴的数组对象进行纵向的拼接（纵向的拼接沿着axis= 1方向）；**注：一般axis = 0，就是对该轴向的数组进行操作，操作方向是另外一个轴，即axis=1。**

```python
In [23]: a = np.array([[1, 2], [3, 4]])

In [24]: b = np.array([[5, 6]])

In [25]: np.concatenate((a, b), axis=0)
Out[25]:
array([[1, 2],
       [3, 4],
       [5, 6]])
```



# np.nonzero

- [参考](https://blog.csdn.net/u013698770/article/details/54632047)

- 返回数组a中非零元素的索引值数组。

（1）只有a中非零元素才会有索引值，那些零值元素没有索引值；
（2）返回的索引值数组是一个2维tuple数组，该tuple数组中包含一维的array数组。其中，一维array向量的个数与a的维数是一致的。
（3）索引值数组的每一个array均是从一个维度上来描述其索引值。比如，如果a是一个二维数组，则索引值数组有两个array，第一个array从行维度来描述索引值；第二个array从列维度来描述索引值。
（4） 该`np.transpose(np.nonzero(x))`函数能够描述出每一个非零元素在不同维度的索引值。
（5）通过`a[nonzero(a)]`得到所有a中的非零值

```python
#a是1维数组
a = [0,2,3]
b = np.nonzero(a)
print(np.array(b).ndim)
print(b)

结果：
2
(array([1, 2], dtype=int64),)
说明：索引1和索引2的位置上元素的值非零。

#a是2维数组
a = np.array([[0,0,3],[0,0,0],[0,0,9]])
b = np.nonzero(a)
print(np.array(b).ndim)
print(b)
print(np.transpose(np.nonzero(a)))
结果：
2
(array([0, 2], dtype=int64), array([2, 2], dtype=int64))
[[0 2]
 [2 2]]
说明：
（1）a中有2个非零元素，因此，索引值tuple中array的长度为2。因为，只有非零元素才有索引值。
（2）索引值数组是2 维的。实际上，无论a的维度是多少，索引值数组一定是2维的tuple，但是tuple中的一维array个数和a的维数一致。
（3）第1个array([0, 2])是从row值上对3和9进行的描述
。第2个array([2, 2])是从col值上对3和9的描述。这样，从行和列上两个维度上各用一个数组来描述非零索引值。
（4）通过调用np.transpose()函数，得出3的索引值是[0 2]
，即第0行，第2列。


#a是3维数组
a = np.array([[[0,1],[1,0]],[[0,1],[1,0]],[[0,0],[1,0]]])
b = np.nonzero(a)
print(np.array(b).ndim)
print(b)
结果：
2
(array([0, 0, 1, 1, 2], dtype=int64), array([0, 1, 0, 1, 1], dtype=int64), array([1, 0, 1, 0, 0], dtype=int64))
说明：由于a是3维数组，因此，索引值数组有3个一维数组。
print(a)
[[[0 1]
  [1 0]]

 [[0 1]
  [1 0]]

 [[0 0]
  [1 0]]]
  a的数组结构如上所示，请将a想像为数量为3的一组小图片，每张图片的大小为2*2，下文中以num * row * col来分别表示其维度。
  b包含3个长度为5的array，这意味着a有3维，且a共有5个非0值。
  先说b中的第1个向量是[0, 0, 1, 1, 2]，这实际是a在num维度上描述的非零值。第0张图上有2个非零值，第1张图上有2个非零值，第2张图上有1个非零值。因此在num维度上的非零值数组为[0, 0, 1, 1, 2]。
  b中的第2个向量是[0, 1, 0, 1, 1]，这实际是a在row维度上描述的非零值。由于row上的值只有0和1（只2行），所以只由0和1组成。
  b中的第3个向量，聪明的读者可能已经明白，不再赘述。
```



# np.mat

- `data`：array_like。输入数据。
- `dtype`：数据类型。输出矩阵的数据类型。

```python
np.mat(data, dtype=None)
不同于np.matrix，如果输入本身就已经是matrix或ndarray ，则np.asmatrix不会复制输入，而是仅仅创建了一个新的引用。

>>> x = np.array([[1, 2], [3, 4]])
>>> m = np.asmatrix(x)
>>> x[0,0] = 5
>>> m
matrix([[5, 2],
        [3, 4]])


```



# np.unique

- [参考](https://blog.csdn.net/qq_33591712/article/details/85095075)

```python
data = np.array([[1,8,3,3,4],
                 [1,8,9,9,4],
                 [1,8,3,3,4]])
 #删除整个数组的重复元素      
uniques = np.unique(data)
print( uniques)
array([1, 3, 4, 8, 9])
 #删除重复行      
uniques = np.unique(data，axis=0)
print( uniques)
array([[1,8,3,3,4],
	 [1,8,9,9,4]])
 #删除重复列
uniques = np.unique(data，axis=1)


```



# np.dot

- [参考](https://blog.csdn.net/weixin_43584807/article/details/103105709)

1. Numpy中dot()函数主要功能有两个：向量点积和矩阵乘法
2. x.dot(y) 等价于 np.dot(x,y) ———x是m*n 矩阵 ，y是n*m矩阵，则x.dot(y) 得到m*m矩阵。
3. 如果处理的是一维数组，则得到的是两数组的內积。
4. 如果处理的是二维数组（矩阵），则得到的是矩阵积
   1. np.dot(x, y), 当x为二维矩阵，y为一维向量，这时y会转换一维矩阵进行计算
   2. np.dot(x, y)中，x、y都是二维矩阵，进行矩阵积计算

```python
import numpy as np
x=np.arange(0,6).reshape(2,3)
y=np.random.randint(0,10,size=(3,2))
print (x)
print (y)
print ("x.shape:"+str(x.shape))
print ("y.shape"+str(y.shape))
print (np.dot(x,y))

[[0 1 2]
 [3 4 5]]
[[7 5]
 [0 7]
 [6 2]]
x.shape:(2, 3)
y.shape(3, 2)
[[12 11]
 [51 53]]

```

# np.logical_and/or/not

```python
# and
np.logical_and(x1, x2, *args, **kwargs)
>>> np.logical_and(True, False)
False
>>> np.logical_and([True, False], [False, False])
array([False, False], dtype=bool)

>>> x = np.arange(5)
>>> np.logical_and(x>1, x<4)
array([False, False,  True,  True, False], dtype=bool)

# or
np.logical_or(x1, x2, *args, **kwargs)
>>> np.logical_or(True, False)
True
>>> np.logical_or([True, False], [False, False])
array([ True, False], dtype=bool)

>>> x = np.arange(5)
>>> np.logical_or(x < 1, x > 3)
array([ True, False, False, False,  True], dtype=bool)

# not
logical_not(x, *args, **kwargs)
>>> np.logical_not(3)
False
>>> np.logical_not([True, False, 0, 1])
array([False,  True,  True, False], dtype=bool)

>>> x = np.arange(5)
>>> np.logical_not(x<3)
array([False, False, False,  True,  True], dtype=bool)
```

# np.transpose

- 变换纬度，通常把 h*w * c 的图片转化为 c h w

```python
def transpose(a, axes=None):
Args:
    a : array_like
        Input array.
    axes : tuple or list of ints, optional
        If specified, it must be a tuple or list which contains a permutation of
        [0,1,..,N-1] where N is the number of axes of a.  The i'th axis of the
        returned array will correspond to the axis numbered ``axes[i]`` of the
        input.  If not specified, defaults to ``range(a.ndim)[::-1]``, which
        reverses the order of the axes.

>>> x = np.arange(4).reshape((2,2))
>>> x
array([[0, 1],
       [2, 3]])

>>> np.transpose(x)
array([[0, 2],
       [1, 3]])

>>> x = np.ones((1, 2, 3))
>>> np.transpose(x, (1, 0, 2)).shape
(2, 1, 3)

```



# np.insert

```python
def insert(arr, obj, values, axis=None)
    >>> a = np.array([[1, 1], [2, 2], [3, 3]])
    >>> a
    array([[1, 1],
           [2, 2],
           [3, 3]])
    >>> np.insert(a, 1, 5)
    array([1, 5, 1, ..., 2, 3, 3])
    >>> np.insert(a, 1, 5, axis=1)
    array([[1, 5, 1],
           [2, 5, 2],
           [3, 5, 3]])

    Difference between sequence and scalars:

    >>> np.insert(a, [1], [[1],[2],[3]], axis=1)
    array([[1, 1, 1],
           [2, 2, 2],
           [3, 3, 3]])
    >>> np.array_equal(np.insert(a, 1, [1, 2, 3], axis=1),
    ...                np.insert(a, [1], [[1],[2],[3]], axis=1))
    True

    >>> b = a.flatten()
    >>> b
    array([1, 1, 2, 2, 3, 3])
    >>> np.insert(b, [2, 2], [5, 6])
    array([1, 1, 5, ..., 2, 3, 3])

    >>> np.insert(b, slice(2, 4), [5, 6])
    array([1, 1, 5, ..., 2, 3, 3])

    >>> np.insert(b, [2, 2], [7.13, False]) # type casting
    array([1, 1, 7, ..., 2, 3, 3])

    >>> x = np.arange(8).reshape(2, 4)
    >>> idx = (1, 3)
    >>> np.insert(x, idx, 999, axis=1)
    array([[  0, 999,   1,   2, 999,   3],
           [  4, 999,   5,   6, 999,   7]])

```

# np.logspace

- 创建等比数列， base^start -> base^stop  num个

```python
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None,
             axis=0)

    >>> y = np.linspace(start, stop, num=num, endpoint=endpoint)
    ... # doctest: +SKIP
    >>> power(base, y).astype(dtype)
    ... # doctest: +SKIP

    Examples
    --------
    >>> np.logspace(2.0, 3.0, num=4)
    array([ 100.        ,  215.443469  ,  464.15888336, 1000.        ])
    >>> np.logspace(2.0, 3.0, num=4, endpoint=False)
    array([100.        ,  177.827941  ,  316.22776602,  562.34132519])
    >>> np.logspace(2.0, 3.0, num=4, base=2.0)
    array([4.        ,  5.0396842 ,  6.34960421,  8.        ])

    Graphical illustration:

    >>> import matplotlib.pyplot as plt
    >>> N = 10
    >>> x1 = np.logspace(0.1, 1, N, endpoint=True)
    >>> x2 = np.logspace(0.1, 1, N, endpoint=False)
    >>> y = np.zeros(N)
    >>> plt.plot(x1, y, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.plot(x2, y + 0.5, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.ylim([-0.5, 1])
    (-0.5, 1)
    >>> plt.show()
```



# np.where

- np.where(condition, x, y)

满足条件(condition)，输出x，不满足输出y。

```python
>>> aa = np.arange(10)
>>> np.where(aa,1,-1)
array([-1,  1,  1,  1,  1,  1,  1,  1,  1,  1])  # 0为False，所以第一个输出-1
>>> np.where(aa > 5,1,-1)
array([-1, -1, -1, -1, -1, -1,  1,  1,  1,  1])

>>> np.where([[True,False], [True,True]],    # 官网上的例子
			 [[1,2], [3,4]],
             [[9,8], [7,6]])
array([[1, 8],
	   [3, 4]])
```


[TOC]



# np.arange

```python
#生成几个相同的数组并合成一个shape = [20,5]
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
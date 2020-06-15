[TOC]

# numpy

## np.arange

```python
#生成几个相同的数组并合成一个shape = [20,5]
[np.arange(5)] * 4 * 5
```



## 双冒号

```python
seq[start:end:step]
```



## np.tile

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



## np.nonzero

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


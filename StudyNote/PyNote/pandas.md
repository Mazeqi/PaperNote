[TOC]



# DataFrame

- 把DataFrame当作一个由若干Series对象构成的字典

```python
 df = pd.DataFrame({'key1':['a', 'a', 'b', 'b', 'a'],
                    'key2':['one', 'two', 'one', 'two', 'one'],
                    'data1':np.random.randn(5),
                    'data2':np.random.randn(5)})
print(df)

      data1     data2 key1 key2
0 -0.410122  0.247895    a  one
1 -0.627470 -0.989268    a  two
2  0.179488 -0.054570    b  one
3 -0.299878 -1.640494    b  two
4 -0.297191  0.954447    a  one

area = pd.Series({'Guangzhou':55555, 'Shenzhen':44444, 'Dongguan':33333, 'Foshan':22222, 'Zhuhai':11111})
pop = pd.Series({'Guangzhou':51, 'Shenzhen':42, 'Dongguan':33, 'Foshan':24, 'Zhuhai':15})
data = pd.DataFrame({'area':area, 'pop':pop})
data
```



# groupby

- [参考](https://blog.csdn.net/u013317445/article/details/85268877)

```python
#demo1----------------------------------------------------------------------
list(df.groupby(['key1']))#list后得到：[(group1),(group2),......]
[('a',       data1     data2 key1 key2
  0 -0.410122  0.247895    a  one
  1 -0.627470 -0.989268    a  two
  4 -0.297191  0.954447    a  one), 
 ('b',       data1     data2 key1 key2
  2  0.179488 -0.054570    b  one
  3 -0.299878 -1.640494    b  two)]
# list后得到：[(group1),(group2),…]
# 每个数据片(group)格式: (name,group)元组

for name,group in df.groupby(['key1']):
    print(name)
    print(group)
'''
a
      data1     data2 key1 key2
0 -0.410122  0.247895    a  one
1 -0.627470 -0.989268    a  two
4 -0.297191  0.954447    a  one
b
      data1     data2 key1 key2
2  0.179488 -0.054570    b  one
3 -0.299878 -1.640494    b  two
'''


#demo2-----------------------------------------------------------------------------
# 对于多重键，产生的一组二元元组：（（k1,k2），数据块）,（（k1,k2），数据块）…
# 第一个元素是由键值组成的元组

for name,group in df.groupby(['key1','key2']):
    print(name)  #name=(k1,k2)
    print(group)
'''
('a', 'one')
      data1     data2 key1 key2
0 -0.410122  0.247895    a  one
4 -0.297191  0.954447    a  one
('a', 'two')
     data1     data2 key1 key2
1 -0.62747 -0.989268    a  two
('b', 'one')
      data1    data2 key1 key2
2  0.179488 -0.05457    b  one
('b', 'two')
      data1     data2 key1 key2
3 -0.299878 -1.640494    b  two
'''
```





# read_csv

- [参考](https://www.jianshu.com/p/9c12fb248ccc)

- read_csv读取的数据类型为Dataframe
- obj.dtypes可以查看每列的数据类型
- header = 0表示文件第0行（即第一行，索引从0开始）为列索引，这样加names会替换原来的列索引。
- index_col:int类型，默认none，把真实序列的某列当作index

```python
pandas.read_csv(‘test.csv’)
```



# astype

```python
# 类型转化
data.astype(int) 
```



# value_cunts

- value_counts()是一种查看表格某列中有多少个不同值的快捷方法，并计算每个不同值有在该列中有多少重复值。
- value_counts()是Series拥有的方法，一般在DataFrame中使用时，需要指定对哪一列或行使用

```python
data['Unit Name'].value_counts()
```



# sample

- n ：要抽取的行数
- frac ：抽取的比例，如frac = 0.8 就是抽取其中的百分之80
- replace：True有放回抽样
- weights：字符索引或改了数组，axis=0为行字符索引或概率数组
- random_state:随机数发生器种子，random_state =None,取得数据不重复，=1可重复

```python
DataFrame.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)[source]
```



# scipy.signal.resample

- [参考](https://vimsky.com/examples/usage/python-scipy.signal.resample.html)

- 使用傅里叶方法沿给定轴将x重新采样为num个采样。

- 重采样的信号从与x相同的值开始，但以`len(x) / num * (spacing of x)`。因为使用了傅里叶方法，所以假定信号是周期性的。

```python
from scipy import signal
'''
x：array_like
要重新采样的数据。
num：int
重采样信号中的采样数。
t：array_like, 可选参数
如果给定t，则假定为x中与信号数据关联的等距采样位置。
axis：int, 可选参数
重新采样的x轴。默认值为0。
window：array_like, callable, string, float, 或 tuple, 可选参数
指定在傅立叶域中应用于信号的窗口。有关详情，请参见下文。
返回值：
resampled_x或(resampled_x，resampled_t)
重新采样的数组，或者，如果给定了t，则包含重新采样的数组和相应的重新采样位置的元组。
'''
scipy.signal.resample(x, num, t=None, axis=0, window=None)
```



# drop

```python
#labels 就是要删除的行列的名字，用列表给定
#axis 默认为0，指删除行，因此删除columns时要指定axis=1；
#index 直接指定要删除的行
#columns 直接指定要删除的列
#inplace=False，默认该删除操作不改变原数据，而是返回一个执行删除操作后的新dataframe；
#inplace=True，则会直接在原数据上进行删除操作，删除后无法返回。

# 因此，删除行列有两种方式：
# 1）labels=None,axis=0 的组合
# 2）index或columns直接指定要删除的行或列

>>>df = pd.DataFrame(np.arange(12).reshape(3,4), columns=['A', 'B', 'C', 'D'])

>>>df

   A   B   C   D

0  0   1   2   3

1  4   5   6   7

2  8   9  10  11

#Drop columns,两种方法等价

>>>df.drop(['B', 'C'], axis=1)

   A   D

0  0   3

1  4   7

2  8  11

>>>df.drop(columns=['B', 'C'])

   A   D

0  0   3

1  4   7

2  8  11

# 第一种方法下删除column一定要指定axis=1,否则会报错
>>> df.drop(['B', 'C'])

ValueError: labels ['B' 'C'] not contained in axis

#Drop rows
>>>df.drop([0, 1])

   A  B   C   D

2  8  9  10  11

>>> df.drop(index=[0, 1])

   A  B   C   D
   
2  8  9  10  11

```



# unique

- 下面代码将取「name」列的唯一实体：

```python
df["name"].unique()
```



# read_excel

```python
 self.data_1_df = pd.read_excel('data/附件一：已结束项目任务数据.xls')
 print(self.data_1_df.info())
 print(self.data_1_df.head(10))
```


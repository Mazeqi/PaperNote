[TOC]

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


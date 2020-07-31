[TOC]

# sklearn.preprocessing

## LabelEncoder

- 简单来说 LabelEncoder 是对不连续的数字或者文本进行按序编号，可以用来生成属性/标签

```python
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
encoder.fit([1,3,2,6])
t=encoder.transform([1,6,6,2])
print(t)

# output
[0 3 3 1]
```



## OneHotEncoder

- OneHotEncoder 用于将表示分类的数据扩维，将[[1]，[2]，[3]，[4]]映射为 0,1,2,3的位置为1（高维的数据自己可以测试）

```python
from sklearn.preprocessing import OneHotEncoder
oneHot=OneHotEncoder()#声明一个编码器
oneHot.fit([[1],[2],[3],[4]])
print(oneHot.transform([[2],[3],[1],[4]]).toarray())

#output
[[0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [1. 0. 0. 0.]
 [0. 0. 0. 1.]
```



## standardization

- (X-X_mean)/X_std 
- 将数据按其属性(按列进行)减去其均值，然后除以其方差。最后得到的结果是，对每个属性/每列来说所有数据都聚集在0附近，方差值为1。
- X：数组或者矩阵
- axis：int类型，初始值为0，axis用来计算均值 means 和标准方差 standard deviations. 如果是0，则单独的标准化每个特征（列），如果是1，则标准化每个观测样本（行）。
- with_mean: boolean类型，默认为True，表示将数据均值规范到0
- with_std: boolean类型，默认为True，表示将数据方差规范到1

```python
sklearn.preprocessing.scale(X, axis=0, with_mean=True,with_std=True,copy=True)

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
X1 = (X-X_mean)/X_std
# use function preprocessing.scale to standardize X
X_scale = preprocessing.scale(X)
```

## StandardScaler

- 该方法也可以对数据X进行标准化处理，实例如下：


```python
from sklearn import preprocessing 
import numpy as np
X = np.array([[ 1., -1.,  2.],
              [ 2.,  0.,  0.],
              [ 0.,  1., -1.]])
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## MinMaxScaler

使用这种方法的目的包括：

- 1、对于方差非常小的属性可以增强其稳定性；
- 2、维持稀疏矩阵中为0的条目。

```python
from sklearn import preprocessing 
import numpy as np
X = np.array([[ 1., -1.,  2.],
              [ 2.,  0.,  0.],
              [ 0.,  1., -1.]])
min_max_scaler = preprocessing.MinMaxScaler()
X_minMax = min_max_scaler.fit_transform(X)

# output
array([[ 0.5       ,  0.        ,  1.        ],
       [ 1.        ,  0.5       ,  0.33333333],
       [ 0.        ,  1.        ,  0.        ]])
```

## Binarization

特征的二值化主要是为了将数据特征转变成boolean变量。在sklearn中，sklearn.preprocessing.Binarizer函数可以实现这一功能。实例如下：

```
>>> X = [[ 1., -1.,  2.],
...      [ 2.,  0.,  0.],
...      [ 0.,  1., -1.]]
>>> binarizer = preprocessing.Binarizer().fit(X)  # fit does nothing
>>> binarizer
Binarizer(copy=True, threshold=0.0)
>>> binarizer.transform(X)
array([[ 1.,  0.,  1.],
       [ 1.,  0.,  0.],
       [ 0.,  1.,  0.]])
```

Binarizer函数也可以设定一个阈值，结果数据值大于阈值的为1，小于阈值的为0，实例代码如下：

```
>>> binarizer = preprocessing.Binarizer(threshold=1.1)
>>> binarizer.transform(X)
array([[ 0.,  0.,  1.],
       [ 1.,  0.,  0.],
       [ 0.,  0.,  0.]])
```

## Imputer

- 由于不同的原因，许多现实中的数据集都包含有缺失值，要么是空白的，要么使用NaNs或者其它的符号替代。这些数据无法直接使用scikit-learn分类器直接训练，所以需要进行处理。幸运地是，sklearn中的**Imputer**类提供了一些基本的方法来处理缺失值，如使用均值、中位值或者缺失值所在列中频繁出现的值来替换。

```python
>>> import numpy as np
>>> from sklearn.preprocessing import Imputer
>>> imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
>>> imp.fit([[1, 2], [np.nan, 3], [7, 6]])
Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)
>>> X = [[np.nan, 2], [6, np.nan], [7, 6]]
>>> print(imp.transform(X))                           
[[ 4.          2.        ]
 [ 6.          3.666...]
 [ 7.          6.        ]]
```



# sklearn.model_selection 

## KFold

- [参考](https://blog.csdn.net/weixin_43685844/article/details/88635492)

- n_splits 表示划分为几块（至少是2）
- shuffle 表示是否打乱划分，默认False，即不打乱
- random_state 表示是否固定随机起点，Used when shuffle == True.

1，get_n_splits([X, y, groups]) 返回分的块数
2，split(X[,Y,groups]) 返回分类后数据集的index

```python
KFold(n_splits=’warn’, shuffle=False, random_state=None)

# demo1
#-------------------------------------------------------------------------------------
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=2)
print( kf.get_n_splits(X))

#output
2

#demo2
#-------------------------------------------------------------------------------------
for train_index, test_index in kf.split(X):
	print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
	y_train, y_test = y.iloc[train_index], y.iloc[test_index]

TRAIN: [2 3] TEST: [0 1]
TRAIN: [0 1] TEST: [2 3]

#demo3
#-------------------------------------------------------------------------------------
for k,(train,test) in enumerate(kf.split(x,y)):
	print (k,(train,test))
	x_train=X.iloc[train]
	x_test=X.iloc[test]
	y_train=Y.iloc[train]
	y_test=Y.iloc[tes]
        
```



## train_test_split

- 将数据集划分为训练集和测试集

```python
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
iris.data.shape, iris.target.shape
((150, 4), (150,))

x 是data y是label
>>> X_train, X_test, y_train, y_test = train_test_split(
...     iris.data, iris.target, test_size=0.4, random_state=0)

>>> X_train.shape, y_train.shape
((90, 4), (90,))
>>> X_test.shape, y_test.shape
((60, 4), (60,))

>>> clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
>>> clf.score(X_test, y_test)                           
0.96...

```



## StratifiedKFold 

- [参考](https://blog.csdn.net/luoganttcc/article/details/106244453)

- StratifiedKFold 将X_train和 X_test 做有放回抽样，随机分三次，取出索引

  ```python
  import numpy as np
  from sklearn.model_selection import StratifiedKFold
  
  # demo1
  #---------------------------------------------------------------------------------
  X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
  y = np.array([0, 0, 1, 1])
  skf = StratifiedKFold(n_splits=2).split(X, y)
  #c= skf.get_n_splits(X, y)
  
  for train_index, test_index in skf:
       print("TRAIN:", train_index, "TEST:", test_index)
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
  
  # output
  TRAIN: [1 3] TEST: [0 2]
  TRAIN: [0 2] TEST: [1 3]
          
  #demo2
  #---------------------------------------------------------------------------------
  import numpy as np
  from sklearn.model_selection import StratifiedKFold
  X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
  y = np.array([0, 0, 1, 1])
  skf = StratifiedKFold(n_splits=2).split(X, y)
  
  print(list(skf))
  #output
  [(array([1, 3]), array([0, 2])), (array([0, 2]), array([1, 3]))]
  
  ```

# TfidfVectorizer

- [参考](https://zhuanlan.zhihu.com/p/67883024) [参数解析](https://blog.csdn.net/laobai1015/article/details/80451371)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)

train = ["Chinese Beijing Chinese",
          "Chinese Chinese Shanghai",
   		  "Chinese Macao",
   		  "Tokyo Japan Chinese"]
    
tv_fit = tv.fit_transform(train)
tv.get_feature_names()

output : ['beijing', 'chinese', 'japan', 'macao', 'shanghai', 'tokyo']
tv_fit.toarray()

Out[8]:
array([[1.91629073, 2.        , 0.        , 0.        , 0.        , 0.        ],
       [0.        , 2.        , 0.        , 0.        , 1.91629073, 0.        ],
       [0.        , 1.        , 0.        , 1.91629073, 0.        , 0.        ],
       [0.        , 1.        , 1.91629073, 0.        , 0.        , 1.91629073]])

# 词语beijing的在第1篇文本中的频次为.0，tf(beijing,d1)=1.0
# 词语beijing只在第1篇文本中出现过df(d,beijing)=1,nd=4,
# 代入平滑版的tf-idf计算式得到1.9
In [13]: 1.0*(1+log((4+1)/(1+1)))
Out[13]: 1.916290731874155
# 词语chinese的在第1篇文本中的频次为2.0，tf(chinese,d1)=2.0
# 词语chinese只在4篇文本中都出现过df(d,beijing)=4,nd=4,
# 代入平滑版的tf-idf计算式得到2.0
In [14]: 2.0*(1+log(4/4))
Out[14]: 2.0
```



# Sklearn.metrics

- [参考1](https://blog.csdn.net/CherDW/article/details/55813071) [参考2](https://blog.csdn.net/Jinyindao243052/article/details/107272171)

## accuracy_score

- 分类准确率分数是指所有分类正确的百分比。分类准确率这一衡量分类器的标准比较容易理解，但是它不能告诉你响应值的潜在分布，并且它也不能告诉你分类器犯错的类型。

```python
sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
# normalize：默认值为True，返回正确分类的比例；如果为False，返回正确分类的样本数

>>>import numpy as np
>>>from sklearn.metrics import accuracy_score
>>>y_pred = [0, 2, 1, 3]
>>>y_true = [0, 1, 2, 3]
>>>accuracy_score(y_true, y_pred)
0.5
>>>accuracy_score(y_true, y_pred, normalize=False)
2
```

## precision_score

$$
precision = \frac{tp}{tp + fp}
$$



```python
from sklearn import metrics
y_pred = [0, 1, 0, 0]
y_true = [0, 1, 0, 1]
metrics.precision_score(y_true, y_pred)
1.0
```

## recall_score

$$
recall = \frac{tp}{tp+fn}
$$



- 召回率 =提取出的正确信息条数 /样本中的信息条数。通俗地说，就是所有准确的条目有多少被检索出来了。 
- average : string, [None, ‘micro’, ‘macro’(default), ‘samples’, ‘weighted’] 将一个二分类matrics拓展到多分类或多标签问题时，我们可以将数据看成多个二分类问题的集合，每个类都是一个二分类。接着，我们可以通过跨多个分类计算每个二分类metrics得分的均值，这在一些情况下很有用。你可以使用average参数来指定。
  - macro：计算二分类metrics的均值，为每个类给出相同权重的分值。当小类很重要时会出问题，因为该macro-averging方法是对性能的平均。另一方面，该方法假设所有分类都是一样重要的，因此macro-averaging方法会对小类的性能影响很大。
  -  weighted:对于不均衡数量的类来说，计算二分类metrics的平均，通过在每个类的score上进行加权实现。
  -  micro：给出了每个样本类以及它对整个metrics的贡献的pair（sample-weight），而非对整个类的metrics求和，它会每个类的metrics上的权重及因子进行求和，来计算整个份额。Micro-averaging方法在多标签（multilabel）问题中设置，包含多分类，此时，大类将被忽略。
  - samples：应用在multilabel问题上。它不会计算每个类，相反，它会在评估数据中，通过计算真实类和预测类的差异的metrics，来求平均（sample_weight-weighted）
- average：average=None将返回一个数组，它包含了每个类的得分.

```python
klearn.metrics.recall_score(y_true, y_pred, labels=None, pos_label=1,average='binary', sample_weight=None)


>>>from sklearn.metrics import recall_score
>>>y_true = [0, 1, 2, 0, 1, 2]
>>>y_pred = [0, 2, 1, 0, 0, 1]
>>>recall_score(y_true, y_pred, average='macro') 
0.33...
>>>recall_score(y_true, y_pred, average='micro') 
0.33...
>>>recall_score(y_true, y_pred, average='weighted') 
0.33...
>>>recall_score(y_true, y_pred, average=None)
array([1.,  0., 0.])
```

## fbeta_score

$$
F_{\beta} = (1 + \beta ^2)\frac{precision \times recall}{\beta^2 precision +recall}
$$



```python
from sklearn import metrics
y_pred = [0, 1, 0, 0]
y_true = [0, 1, 0, 1]
metrics.precision_score(y_true, y_pred)
1.0
metrics.recall_score(y_true, y_pred)
0.5
metrics.f1_score(y_true, y_pred)
0.66...
metrics.fbeta_score(y_true, y_pred, beta=0.5)
0.83...
```

## fowlkes_mallows_score

$$
FMI = \frac{TP}{\sqrt{(TP+FP)(TP+FN)}}
$$

```python
from sklearn import metrics
labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [0, 0, 1, 1, 2, 2]
print(metrics.fowlkes_mallows_score(labels_true, labels_pred))
```

## balanced_accuracy_score 

$$
balanced-accuracy = \frac{1}{2}(\frac{TP}{TP+FN} + \frac{TN}{TN+FP})
$$

```python
from sklearn.metrics import balanced_accuracy_score
y_true = [0, 1, 0, 0, 1, 0]
y_pred = [0, 1, 0, 0, 0, 1]
balanced_accuracy_score(y_true, y_pred)
```

## average_precision_score

$$
AP = \sum_{n}(R_n - R_{n-1})P_n
$$

```python
from sklearn.metrics import average_precision_score
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
average_precision_score(y_true, y_scores)
```



## roc_curve

- ROC观察模型正确地识别正例的比例与模型错误地把负例数据识别成正例的比例之间的权衡。TPR的增加以FPR的增加为代价。**ROC曲线下的面积是模型准确率的度量，AUC（Area under roccurve）。**

- ***纵坐标******：******真正率***（True Positive Rate , TPR）或灵敏度（sensitivity）

  - TPR = TP /(TP + FN） （正样本预测结果数 / 正样本实际数）

- **横坐标**：假正率（False Positive Rate , FPR）

  - *FPR = FP /（FP + TN） （被预测为正的负样本结果数 /负样本实际数）*

- **Return**

  - `y_true` : array, **shape = [n_samples]** True binary labels. **If labels are not either {-1, 1} or {0, 1}, then pos_label should be explicitly given**
  - `y_score` : array, **shape = [n_samples]**
  - `pos_label` : `int or str, default=None` , **Label considered as positive and others are considered negative.**

  ```python
  # pos_label = 1即表示标签为1的是正样本，其余的都是负样本，因为这个只能做二分类。
  sklearn.metrics.roc_curve(y_true,y_score, pos_label=None, sample_weight=None, drop_intermediate=True)
  
  # demo1
  # --------------------------------------------------------------------------
  >>>import numpy as np
  >>>from sklearn import metrics
  >>>y = np.array([1, 1, 2, 2])
  >>>scores = np.array([0.1, 0.4, 0.35, 0.8])
  >>>fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
  >>>fpr
  array([0. ,  0.5,  0.5, 1. ])
  >>>tpr
  array([0.5,  0.5,  1. , 1. ])
  >>>thresholds
  array([0.8 ,  0.4 ,  0.35, 0.1 ])
  >>>from sklearn.metrics import auc 
  >>>metrics.auc(fpr, tpr) 
  0.75 
  
  # demo2
  # --------------------------------------------------------------------------------
  import numpy as np
  from sklearn import metrics
  import matplotlib.pyplot as plt
  from sklearn.metrics import auc
  
  y = np.array([1,1,2,3])
  #y为数据的真实标签
  scores = np.array([0.1, 0.2, 0.35, 0.8])
  
  #scores为分类其预测的得分
  fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
  print(auc(fpr,tpr))
  
  #得到fpr,tpr, thresholds
  plt.plot(fpr,tpr,marker = 'o')
  plt.show()
  ```

## roc_auc_score

- **Parameters**

  - `y_true` : array, **shape = [n_samples] or [n_samples, n_classes]**
  - `y_score` : array, **shape = [n_samples] or [n_samples, n_classes]**
  - `average` : string, **[None, ‘micro’, ‘macro’ (default), ‘samples’, ‘weighted’]**，If None, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data。

- #### Returns:

  - `auc` : float

  ```python
  ### roc_auc_score
  import numpy as np
  from sklearn.metrics import roc_auc_score
  y_true = np.array([0, 0, 1, 1])
  y_scores = np.array([0.1, 0.4, 0.35, 0.8])
  roc_auc_score(y_true, y_scores)
  
  0.75
  # roc_auc_score 是 预测得分曲线下的 auc，在计算的时候调用了 auc；
  def _binary_roc_auc_score(y_true, y_score, sample_weight=None):
  	if len(np.unique(y_true)) != 2:
  		raise ValueError("Only one class present in y_true. ROC AUC score "
  "is not defined in that case.")
  
  	fpr, tpr, tresholds = roc_curve(y_true, y_score,
  	sample_weight=sample_weight)
  	return auc(fpr, tpr, reorder=True)
  ```

  

# jieba

- [地址](https://github.com/fxsjy/jieba)

```python
# encoding=utf-8
import jieba

jieba.enable_paddle()# 启动paddle模式。 0.40版之后开始支持，早期版本不支持
strs=["我来到北京清华大学","乒乓球拍卖完了","中国科学技术大学"]
for str in strs:
    seg_list = jieba.cut(str,use_paddle=True) # 使用paddle模式
    print("Paddle Mode: " + '/'.join(list(seg_list)))

seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
print(", ".join(seg_list))

seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
print(", ".join(seg_list))


output
【全模式】: 我/ 来到/ 北京/ 清华/ 清华大学/ 华大/ 大学

【精确模式】: 我/ 来到/ 北京/ 清华大学

【新词识别】：他, 来到, 了, 网易, 杭研, 大厦    (此处，“杭研”并没有在词典中，但是也被Viterbi算法识别出来了)

【搜索引擎模式】： 小明, 硕士, 毕业, 于, 中国, 科学, 学院, 科学院, 中国科学院, 计算, 计算所, 后, 在, 日本, 京都, 大学, 日本京都大学, 深造
```


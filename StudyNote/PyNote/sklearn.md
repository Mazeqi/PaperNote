[TOC]

# train_test_split

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





# jieba

- [地址](https://github.com/fxsjy/jieba)

## demo

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


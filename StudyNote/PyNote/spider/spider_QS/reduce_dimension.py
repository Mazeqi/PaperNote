from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import umap
import matplotlib.pyplot as plt
from numpy import *
import operator
import json
import codecs

'''
    args:
        if_show: 是否输出降维后的二维数据图
        newInput：要输入的词，这里只能为词典中的词，用于推荐相似的词
        k：k个近邻
'''
def recommend(if_show = False, newInput = "", k = 3):
    # 作业2存储的词向量
    OUTPUT_VEC = r"content\word2vec.bin"
    
    # json索引
    WORD_INDEX = r"content/wordindex.json"

    RECOMEND_WORD = r"content/recommendWords.txt"

    model = KeyedVectors.load_word2vec_format(OUTPUT_VEC, binary=True)

    # extract all vectors
    X = model.vectors
    #print(model.vector_size)

    '''
    n_neighbors: float (optional, default 15)
       用于流形近似的局部邻域的大小(根据邻域抽样点的数目)。
       值越大，流形的全局视图就越多，而值越小，保留的局部
       数据就越多。一般取值范围在2到100之间。

    n_components: int (optional, default 2)
      要嵌入的空间尺寸。默认值为2以方便可视化，
      但是可以合理地设置为2到100范围内的任何整数值。

    min_dist: float (optional, default 0.1)
    嵌入点之间的有效最小距离。较小的值将导致更聚簇/聚集嵌入，
    在流形上附近的点被拉近到一起，而较大的值将导致更均匀的点
    分散。该值应相对于“spread”值进行设置，该值决定嵌入点展开的范围。

    '''

    model_embedding = umap.UMAP(n_neighbors=30, min_dist=0.1,
                                n_components=2)
    cluster_embedding = model_embedding.fit_transform(X)

    #print(cluster_embedding.shape)

    if if_show:
        plt.figure(figsize=(10,9))
        plt.scatter(cluster_embedding[:, 0], cluster_embedding[:, 1], s=3, cmap='Spectral')
        plt.show()
    

    '''
        KNN（K-Nearest Neighbor algorithm），K最近邻算法，
        通过计算样本个体间的距离或者相似度寻找与每个样本个体最相近的K个个体，
        算法的时间复杂度跟样本的个数直接相关，需要完成一次两两比较的过程。
        KNN一般被用于分类算法，在给定分类规则的训练集的基础上对总体的样本进行分类，
        是一种监督学习（Supervised learning）方法。

        这里我们不用KNN来实现分类，我们使用KNN最原始的算法思路，即为每个内容寻找K个与其最相似的内容，
        并推荐给用户。相当于每个内容之间都会完成一次两两比较的过程，如果你的网站有n个内容，
        那么算法的时间复杂度为Cn2，即n（n-1）/2。

    '''
    # 加载所有的词
    fin = codecs.open(WORD_INDEX, 'r', encoding='utf-8')
    wordIndex = json.load(fin)

    indexWord = {}
    for word,index in wordIndex.items():
        indexWord[index] = word

    # shape[0]表示行数
    numSamples = cluster_embedding.shape[0]   

    # step1：计算距离
    # tile(A, reps): 构造一个矩阵，通过A重复reps次得到
    # the following copy numSamples rows for dataSet
     # 按元素求差值
    newInput = cluster_embedding[wordIndex[newInput]]
    diff = tile(newInput, (numSamples, 1)) - cluster_embedding 

    # 将差值平方
    squaredDiff = diff ** 2 

    # 按行累加
    squaredDist = sum(squaredDiff, axis = 1)  

    # 将差值平方和求开方，即得距离
    distance = squaredDist ** 0.5 

    # # step 2: 对距离排序
    # argsort() 返回排序后的索引值
    sortedDistIndices = argsort(distance)

    k_recommend = []
    fout = codecs.open(RECOMEND_WORD, "w", encoding="utf8")
    for i in range(k):
        # # step 3: 选择k个最近邻
        word = indexWord[sortedDistIndices[i]]

        k_recommend.append(word)

        fout.write(word + " ")
        if i % 10 == 0:
            fout.write("\n")

    fout.close()

    return k_recommend

if __name__ == "__main__":
    words = recommend(if_show = False, newInput = "梦若", k = 50)
    print(words)
import jieba
import codecs
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import json
import collections
from gensim.models import KeyedVectors
from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer

# 爬取的内容
INPUT_RAW = r"content/QS_content.txt"

# 对爬取的内容进行分词
OUTPUT_FENCI = r'content/QS_jieba.txt'

# 分词之后进行向量计算
OUTPUT_VEC = r"content\word2vec.bin"

# 产生json索引
WORD_INDEX = r"content/wordindex.json"

# 产生annoy索引
INDEX_SAVE = r"content/annoy.index"

# k-近邻新词
NEW_WORD_SAVE = r"content/new_word.txt"

# tf_df 新词
NEW_WORD_Tf_SAVE = r"content/new_tf_word.txt"
'''
    步骤一 对文本内容进行分词
'''
def DivideWord():
    fin = codecs.open(INPUT_RAW, "r", encoding="utf8") 
    fout = codecs.open(OUTPUT_FENCI, "w", encoding="utf8")

    for line in fin:
        if line.isspace():
            continue

        cut_line = list(jieba.cut(line))
        line_str = " ".join(cut_line)
        fout.write(line_str)
        
    fin.close()
    fout.close()

'''
    步骤二 对分词形成词向量
'''
def OutVec():
    fin = codecs.open(OUTPUT_FENCI, "r", encoding="utf8")

    '''

    1.sentences：可以是一个List，对于大语料集，建议使用BrownCorpus,Text8Corpus或·ineSentence构建。
    2.sg： 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。
    3.vector_size：是指输出的词的向量维数，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
    4.window：为训练的窗口大小，8表示每个词考虑前8个词与后8个词（实际代码中还有一个随机选窗口的过程，窗口大小<=5)，默认值为5。
    5.alpha: 是学习速率
    6.seed：用于随机数发生器。与初始化词向量有关。
    7.min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5。
    8.max_vocab_size: 设置词向量构建期间的RAM限制。如果所有独立单词个数超过这个，
        则就消除掉其中最不频繁的一个。每一千万个单词需要大约1GB的RAM。设置成None则没有限制。
    9.sample: 表示 采样的阈值，如果一个词在训练样本中出现的频率越大，那么就越会被采样。
    10.workers:参数控制训练的并行数。多线程
    11.hs: 是否使用HS方法，0表示不使用，1表示使用 。默认为0
    12.negative: 如果>0,则会采用negativesamp·ing，用于设置多少个noise words
    13.cbow_mean: 如果为0，则采用上下文词向量的和，如果为1（default）则采用均值。只有使用CBOW的时候才起作用。
    14.hashfxn： hash函数来初始化权重。默认使用python的hash函数
    15.iter： 迭代次数，默认为5。
    16.trim_rule： 用于设置词汇表的整理规则，指定那些单词要留下，哪些要被删除。
        可以设置为None（min_count会被使用）或者一个接受()
        并返回RU·E_DISCARD,uti·s.RU·E_KEEP或者uti·s.RU·E_DEFAU·T的函数。
    17.sorted_vocab： 如果为1（defau·t），则在分配word index 的时候会先对单词基于频率降序排序。
    18.batch_words：每一批的传递给线程的单词的数量，默认为10000
    '''
    w2v = Word2Vec(sg=1, sentences=LineSentence(fin), min_count=1, vector_size=50, epochs=2 ,window=5)
    w2v.wv.save_word2vec_format(OUTPUT_VEC, binary=True)
    fin.close()


'''
    步骤三 对词向量进行json索引
'''
def JsonIndex():
    model = KeyedVectors.load_word2vec_format(OUTPUT_VEC, binary=True)
    fout = codecs.open(WORD_INDEX, 'w', encoding='utf-8')
    wordindex = collections.OrderedDict()
    for index,key in enumerate(model.key_to_index):
        wordindex[key] = index
        
    json.dump(wordindex, fout, indent=4, ensure_ascii=False) #中文不使用默认的ensure_ascii编码
    fout.close()


'''
    步骤4  对词向量进行annoy索引
    步骤三和四是可以并列的 不相互影响 都是建立索引
'''
def Json2AnnoyIndex():
    model = KeyedVectors.load_word2vec_format(OUTPUT_VEC, binary=True)
    annoyIndex = AnnoyIndex(50, 'angular')  #索引50维向量
    #加载index和向量映射，添加到annoyIndex
    index = 0
    for key in model.index_to_key:
        annoyIndex.add_item(index, model[key])
        index += 1

    annoyIndex.build(4)  #tree_num设置为4，在内存允许的情况下，越大越好
    annoyIndex.save(INDEX_SAVE)    

'''
    测试annoy
'''
def RunAnnoy():
    fin = codecs.open(WORD_INDEX, 'r', encoding='utf-8')
    wordIndex = json.load(fin)
    indexWord = {}
    for word,index in wordIndex.items():
        indexWord[index] = word

    annoyIndex2 = AnnoyIndex(50, 'angular')  
    annoyIndex2.load(INDEX_SAVE)

    '''
    a.get_nns_by_item（i，n，search_k = -1， include_distances = False）
    返回第i 个item的n个最近邻的item。在查询期间，
    它将最多检查search_k个节点（如果未提供，则默认为n_trees * n个）。
    search_k为您提供了更好的准确性和速度之间的运行时权衡。如果将include_distances设置为True，
    它将返回一个包含两个列表的2元素元组：第二个包含所有对应的距离。
    '''
    indexs = annoyIndex2.get_nns_by_item(i = wordIndex['梦若'], n = 20)
    for index in indexs:
        print(index)
        print(indexWord[index])
    
    fin.close()

'''
    得到新词集合
'''
def get_new_word():
    fin = codecs.open(WORD_INDEX, 'r', encoding='utf-8')
    wordIndex = json.load(fin)
    indexWord = {}
    for word,index in wordIndex.items():
        indexWord[index] = word
    fin.close()

    annoyIndex2 = AnnoyIndex(50, 'angular')  
    annoyIndex2.load(INDEX_SAVE)

    '''
    a.get_nns_by_item（i，n，search_k = -1， include_distances = False）
    返回第i 个item的n个最近邻的item。在查询期间，
    它将最多检查search_k个节点（如果未提供，则默认为n_trees * n个）。
    search_k为您提供了更好的准确性和速度之间的运行时权衡。如果将include_distances设置为True，
    它将返回一个包含两个列表的2元素元组：第二个包含所有对应的距离。
    '''

    fout = codecs.open(NEW_WORD_SAVE, "w", encoding="utf8")
    for word,index in wordIndex.items():
        indexs = annoyIndex2.get_nns_by_item(i = wordIndex[word], n = 10)
        new_line = [indexWord[i] for i in indexs]
        print(new_line)
        line_str = " ".join(new_line)
        fout.write(line_str)
    fout.close

'''
TF（词频）的计算很简单，就是针对一个文件t，某个单词Nt 出现在该文档中的频率。
比如文档“I love this movie”，单词“love”的TF为1/4。如果去掉停用词“I”和”it“，则为1/2。

IDF（逆向文件频率）的意义是，对于某个单词t，凡是出现了该单词的文档数Dt，占了全部测试文档D的比例，
再求自然对数。比如单词“movie“一共出现了5次，而文档总数为12，因此IDF为ln(5/12)

'''
def get_DF_TF():
    min_tf_df = 0.0001

    with open(NEW_WORD_SAVE, 'r', encoding='utf8') as f:
        res = f.read()
    
    vector = TfidfVectorizer()
    tf_idf = vector.fit_transform([res])

    word_list = vector.get_feature_names()      # 获取词袋模型的所有词
    weight_list = tf_idf.toarray()

    fout = codecs.open(NEW_WORD_Tf_SAVE, "w", encoding="utf8")
    # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    for i in range(len(weight_list)):
        print("-------第", i+1, "段文本的词语tf-idf权重------")
        for j in range(len(word_list)):
            print(word_list[j], weight_list[i][j])

            if weight_list[i][j] >= min_tf_df:
                line_str = word_list[j] + '\n'
                fout.write(line_str)
    fout.close


    
if __name__ == "__main__":

    # 一个步骤一个步骤执行

    # 分词
    #DivideWord()

    # 词向量
    #OutVec()

    # 生成json 索引
    #JsonIndex()

    # 生成annoy索引
    #AnnoyIndex()

    # 测试annoy
    #RunAnnoy()

    # 得到k近邻新词
    #get_new_word()

    # 得到满足df-idf条件的新词
    get_DF_TF()
import jieba
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as image

#导入小说内容
text_path = 'fiction_content/fiction.txt'

# 将封面做背景
img = np.array(image.open('Img/picture.jpg'))

wc = WordCloud(background_color='white',  # 背景颜色
        max_words=1000,  # 最大词数
        mask=img,  # 以该参数值作图绘制词云，这个参数不为空时，width和height会被忽略
        max_font_size=100,  # 显示字体的最大值
        font_path='C:\Windows\Fonts\simfang.ttf',
        width=1280,  # 图片的宽
        height=1024  #图片的长
        )
  
text = open(text_path, encoding='utf-8').read()
wc.generate(text)

# 显示图片
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()
wc.to_file('fiction.png')


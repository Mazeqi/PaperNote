from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as image
import cv2

# 参考：https://blog.csdn.net/FontThrone/article/details/72775865

# 设置停用词
stopwords = set(STOPWORDS)
stopwords.add("糗事")

# 将糗事百科图片做背景
img = np.array(image.open('qiushi.jpg'))
print(img)
img = np.where(img < 255, 0, 255)

wc = WordCloud(background_color='white',  # 背景颜色
        max_words=1000,  # 最大词数
        mask=img,  # 以该参数值作图绘制词云，这个参数不为空时，width和height会被忽略
        max_font_size=100,  # 显示字体的最大值
        font_path='C:\Windows\Fonts\simfang.ttf',
        width=1280,  # 图片的宽
        height=1024,  #图片的长
        stopwords=stopwords
        )
  
text = open('content/QS_content.txt', encoding='utf-8').read()
wc.generate(text)

# 显示图片
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()
wc.to_file('word_cloud.png')


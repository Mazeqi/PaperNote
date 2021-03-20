#conding:utf-8
import requests  #导入库
from bs4 import BeautifulSoup #导入bs
import re
import sys
import os
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED


'''
    return: 每一章的链接和标题
    arg：(url, chapter_name)
'''
def get_novel_chapters():  
    root_url="http://www.botaodz.com/1702.html"

    #发送request请求
    r = requests.get(root_url) 

    #网页编码为utf-8
    r.encoding="utf-8" 

    #创建bs4对象，解析器用官方的html.parser
    soup = BeautifulSoup(r.text,"html.parser")

    #创建一个列表保存每一章的链接
    data = []

    # 保存小说的封面
    div_img = soup.find('div', class_ = 'novelinfo-r')
    img_url = div_img.find('img')['src']
    html = requests.get(img_url)
    with open('Img/picture.jpg', 'wb') as file:
         file.write(html.content)


    #匹配所有li标签
    for li in soup.find_all("li",):
        link = li.find("a",target="_blank",href=re.compile("read"))
        #匹配所有的href带read字样的链接
        if not link:  #如果没获取到链接，就忽略
            continue 
        data.append(("http://www.botaodz.com%s"%link['href'],link.get_text()))

    
    #返回小说所有章节的链接和标题，link.get_text()
    return data  
 

'''
    从所有小说的链接中解析每一章链接中的小说正文部分和标题
    return: content of chapter
'''
def get_chapter_content(url): 
    r = requests.get(url)
    r.encoding="utf-8"
    soup = BeautifulSoup(r.text,"html.parser")

    #获取div标签里的文本get.text()
    return soup.find("div",class_='content').get_text() 


#保存小说内容
if not os.path.exists('fiction_content'):
    os.makedirs('fiction_content')

# 保存封面
if not os.path.exists('Img'):
    os.makedirs('Img')


# 控制一下章的数量
ctr_chapter_len = 300

for chapter in get_novel_chapters():

    if ctr_chapter_len < 0 :
        break
    else:
        ctr_chapter_len -= 1

    #从data文件中读取每一章小说的链接和标题并赋值
    url,title = chapter  
    print(chapter)
    
    #写入小说文本文件名就是标题名
    #encoding要为utf8 否则无输入
    with open("fiction_content/fiction.txt", mode="a", encoding='utf-8') as fout:
        #先写入标题
        fout.write(chapter[-1] + '\n')

        #写入内容
        fout.write(get_chapter_content(url))
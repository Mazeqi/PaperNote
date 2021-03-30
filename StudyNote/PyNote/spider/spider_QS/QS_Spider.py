import requests  
from bs4 import BeautifulSoup 
import sys
import os
import re
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

#发送request请求
user_agent = 'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36'
headers = {'User_agent': user_agent}

#返回每一条糗事百科段子的链接
def get_url_QS(page_num):  
    root_url = "https://www.qiushibaike.com/text/page/" + str(page_num)
    link_url = "https://www.qiushibaike.com"

    r = requests.get(root_url, headers = headers) 

    #utf-8
    r.encoding = "utf-8" 

    #解析器用官方的parser
    soup = BeautifulSoup(r.text,"html.parser")

    # 返回链接列表
    link_url_list = []

    div_top = soup.find('div', class_ = 'col1 old-style-col1')

    for div in div_top.find_all("div"):
        link = div.find("a",target="_blank", class_ = 'contentHerf')

        if link is None:
            continue

        link_url_list.append(link_url + link['href'])

    print(link_url_list)
    return link_url_list  
 

# 返回糗事百科的内容
def get_url_content(url, content_list = None): 

    url_content = {}

    r = requests.get(url, headers = headers)
    r.encoding="utf-8"
    soup = BeautifulSoup(r.text,"html.parser")

    url_content['title']   = soup.find("h1", class_='article-title').get_text() 
    url_content['content'] = soup.find("div", class_='content').get_text()  
    
    content_list.append(url_content)

    if content_list is None:
        return url_content
    else:
        return content_list


# 多线程爬取，参考老师的demo
def crawl_thread_list(num_page):

    urls = []
    for page_num in range(num_page):
        urls.extend(get_url_QS(page_num + 1))
    print(urls)
    content_list = []

    executor = ThreadPoolExecutor(max_workers=20)  
    
    # submit()的参数： 第一个为函数， 之后为该函数的传入参数，允许有多个
    future_tasks = [executor.submit(get_url_content, url, content_list) for url in urls]
    
    # 等待所有的线程完成，才进入后续的执行
    wait(future_tasks, return_when=ALL_COMPLETED)

    return content_list

#保存小说内容
if not os.path.exists('content'):
    os.makedirs('content')

thread_tag = True
if thread_tag:
    for content in crawl_thread_list(13):
        print(content['title'])
        with open("content/QS_content.txt", mode="a", encoding='utf-8') as fout:

            # 写入标题
            fout.write(content['title'] + '\n')

            #写入内容
            fout.write(content['content'] + '\n\n')
else:
    for page_num in range(13):
        for link in get_url_QS(page_num + 1):
            content = get_url_content(link)
            with open("content/QS_content.txt", mode="a", encoding='utf-8') as fout:

                # 写入标题
                fout.write(content['title'] + '\n')

                #写入内容
                fout.write(content['content'] + '\n\n')

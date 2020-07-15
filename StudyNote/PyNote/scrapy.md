# spider demo



## DoubanTop250

- [参考1]([https://blog.csdn.net/qq_1290259791/article/details/82225450#scrapy%E7%88%AC%E5%8F%96%E8%B1%86%E7%93%A3%E7%94%B5%E5%BD%B1](https://blog.csdn.net/qq_1290259791/article/details/82225450#scrapy爬取豆瓣电影)) 

```python
import scrapy
from movie.Item.DbItem import DbItems 

class DoubanSpider(scrapy.Spider):
    name = 'douban'
    allowed_domains = ['movie.douban.com']
    start_urls = ['https://movie.douban.com/top250']
    
    custom_settings = {

        'DEFAULT_REQUEST_HEADERS' :{
            'Referer': 'https://movie.douban.com/top250',
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25'
        }


    }

    def parse(self, response):
        movie_list = response.xpath('//div[@class="article"]//ol[@class="grid_view"]/li')

        for i_item in movie_list:
            # 导入item
            Db_item = DbItems()

            Db_item['movie_index']   = i_item.xpath('.//div[@class="item"]//em/text()').extract_first()

            Db_item['movie_name']  = i_item.xpath( ".//div[@class='info']/div[@class='hd']/a/span[1]/text()").extract_first()

            Db_item['movie_score'] = i_item.xpath('.//span[@class="rating_num"]/text()').extract_first()

            Db_item['comment_num'] = i_item.xpath('.//div[@class="star"]/span[last()]/text()').extract_first()

            # 整一个生成器
            yield Db_item
        next_page = response.xpath('//span[@class="next"]/link/@href').extract_first()

        if next_page:
            yield scrapy.Request("https://movie.douban.com/top250" + next_page, callback = self.parse)
```


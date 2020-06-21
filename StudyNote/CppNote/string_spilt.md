# C++ string_spilt

- [参考](https://www.jianshu.com/p/5876a9f49413)

## find

```C++
size_t find (const string& str, size_t pos = 0) const;
```

- 查找子字符串第一次出现的位置。
- str为子字符串，pos为初始查找位置。
- 返回值：找到的话返回第一次出现的位置，否则返回string::npos。



## substr

```C++
string substr (size_t pos = 0, size_t len = npos) const;
```

- 功能：在原字符串中截取子字符串。
- 参数说明：pos为起始位置，len为要截取子字符串的长度。
- 返回值：子字符串。



## demo1

```C++
vector<string> split(const string &str, const string &pattern)
{
    vector<string> res;
    if(str == "")
        return res;
    //在字符串末尾也加入分隔符，方便截取最后一段
    string strs = str + pattern;
    size_t pos = strs.find(pattern);

    while(pos != strs.npos)
    {
        string temp = strs.substr(0, pos);
        res.push_back(temp);
        //去掉已分割的字符串,在剩下的字符串中进行分割
        strs = strs.substr(pos+1, strs.size());
        pos = strs.find(pattern);
    }

    return res;
}
```



## strtok

```C++
char * strtok ( char * str, const char * delimiters );
```

- 功能：分割字符串str，delimiters为指定的分割符，可以有多个。
- 说明：strtok只能接受C风格的字符串，如果是string类型，可以使用c_str函数进行转换。strtok()用来将字符串分割成一个个片段。参数s指向欲分割的字符串，参数delim则为分割字符串，当strtok()在参数s的字符串中发现到参数delim的分割字符时 则会将该字符改为\0 字符。在第一次调用时，strtok()必需给予参数s字符串，往后的调用则将参数s设置成NULL。每次调用成功则返回被分割出片段的指针。

## demo2

```c
vector<string> split2(const string &str, const string &pattern)
{
    char * strc = new char[strlen(str.c_str())+1];
    strcpy(strc, str.c_str());   //string转换成C-string
    vector<string> res;
    char* temp = strtok(strc, pattern.c_str());
    while(temp != NULL)
    {
        res.push_back(string(temp));
        temp = strtok(NULL, pattern.c_str());
    }
    delete[] strc;
    return res;
}
```



## stringstream

- stringstream为字符串输入输出流，继承自iostream，灵活地使用stringstream流可以完成很多字符串处理功能，例如字符串和其他类型的转换，字符串分割等。在这里，我们使用其实现字符串分割功能。注意stingstream的使用需要包含sstream头文件。

## demo3 

```c
vector<string> split3(const string &str, const char pattern)
{
    vector<string> res;
    stringstream input(str);   //读取str到字符串流中
    string temp;
    //使用getline函数从字符串流中读取,遇到分隔符时停止,和从cin中读取类似
    //注意,getline默认是可以读取空格的
    while(getline(input, temp, pattern))
    {
        res.push_back(temp);
    }
    return res;
}
```


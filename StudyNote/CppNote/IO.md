[TOC]

# stream

- [参考1](https://blog.csdn.net/kingstar158/article/details/6859379)
- [参考2](https://www.cnblogs.com/zhoug2020/p/8427277.html)

- ofstream 是从内存到硬盘，ifstream是从硬盘到内存，其实所谓的缓冲流就是内存

![](./img/iostream.gif)





```C++
#include <fstream>
ofstream         //文件写操作 内存写入存储设备 
ifstream         //文件读操作，存储设备读区到内存中
fstream          //读写操作，对打开的文件可进行读写操作 
```



## open

- 在fstream类中，成员函数open（）实现打开文件的操作，从而将数据流和文件进行关联，通过ofstream,ifstream,fstream对象进行对文件的读写操作

```C++
public member function
 
void open ( const char * filename,
            ios_base::openmode mode = ios_base::in | ios_base::out );
 
void open(const wchar_t *_Filename,
        ios_base::openmode mode= ios_base::in | ios_base::out,
        int prot = ios_base::_Openprot)；
 
```

- **Args**
  - filename  操作文件名
  - mode       打开的方式
  - prot          打开文件的属性

| 方式        | 解释                         |
| ----------- | ---------------------------- |
| os::in      | 为输入(读)而打开文件         |
| ios::out    | 为输出(写)而打开文件         |
| ios::ate    | 初始位置：文件尾             |
| ios::app    | 所有输出附加在文件末尾       |
| ios::trunc  | 如果文件已存在则先删除该文件 |
| ios::binary | 二进制方式                   |

- demo

  ```C++
  ofstream out;
  out.open("Hello.txt", ios::in|ios::out|ios::binary)       
  ```

  

| prot（文件属性） |                    |
| ---------------- | ------------------ |
| 0                | 普通文件，打开操作 |
| 1                | 只读文件           |
| 2                | 隐含文件           |
| 4                | 系统文件           |

- 很多程序中，可能会碰到ofstream out("Hello.txt"), ifstream in("..."),fstream foi("...")这样的的使用，并没有显式的去调用open（）函数就进行文件的操作，直接调用了其默认的打开方式，因为在stream类的构造函数中调用了open()函数,并拥有同样的构造函数，所以在这里可以直接使用流对象进行文件的操作，默认方式如下：

  ```C++
  ofstream out("...", ios::out);
  ifstream in("...", ios::in);
  fstream foi("...", ios::in|ios::out);
  ```

  

## close

- 当文件读写操作完成之后，我们必须将文件关闭以使文件重新变为可访问的。成员函数close()，它负责将缓存中的数据排放出来并关闭文件。这个函数一旦被调用，原先的流对象就可以被用来打开其它的文件了，这个文件也就可以重新被其它的进程所访问了。为防止流对象被销毁时还联系着打开的文件，析构函数将会自动调用关闭函数close。




## write-read

- 类ofstream, ifstream 和fstream 是分别从ostream, istream 和iostream 中引申而来的。这就是为什么 fstream 的对象可以使用其父类的成员来访问数据。
- 一般来说，我们将使用这些类与同控制台(console)交互同样的成员函数(cin 和 cout)来进行输入输出。如下面的例题所示，我们使用重载的插入操作符<<：

- **demo - write** 

  ```C++
  // writing on a text file
  #include <fiostream.h>
  int main () {
      ofstream out("out.txt");
      if (out.is_open()) 
      {
          out << "This is a line.\n";
          out << "This is another line.\n";
          out.close();
      }
      return 0;
  }
  //结果: 在out.txt中写入：
  This is a line.
  This is another line 
  ```

- **demo-read**

```C++
// reading a text file
#include <iostream.h>
#include <fstream.h>
#include <stdlib.h>

int main () {
    char buffer[256];
    ifstream in("test.txt");
    if (! in.is_open())
    { cout << "Error opening file"; exit (1); }
    while (!in.eof() )
    {
        in.getline (buffer,100);
        cout << buffer << endl;
    }
    return 0;
}
//结果 在屏幕上输出
This is a line.
This is another line
```

- 我们使用了一个新的成员函数叫做eof ，它是ifstream 从类 ios 中继承过来的，当到达文件末尾时返回true 。

- 除了eof()以外，还有一些验证流的状态的成员函数（所有都返回bool型返回值）：

  - bad()
    - 如果在读写过程中出错，返回 true 。例如：当我们要对一个不是打开为写状态的文件进行写入时，或者我们要写入的设备没有剩余空间的时候。
  - fail()
    - 除了与bad() 同样的情况下会返回 true 以外，加上格式错误时也返回true ，例如当想要读入一个整数，而获得了一个字母的时候。
  - eof()
    - 如果读文件到达文件末尾，返回true。
  - good()
    - 这是最通用的：如果调用以上任何一个函数返回true 的话，此函数返回 false 。

- 文件流包括两个为顺序读写数据特殊设计的成员函数：write 和 read。

  -  第一个函数 (write) 是ostream 的一个成员函数，都是被ofstream所继承。

  -  而read 是istream 的一个成员函数，被ifstream 所继承。类 fstream 的对象同时拥有这两个函数。它们的原型是：

    ```C++
    write ( char * buffer, streamsize size );
    read ( char * buffer, streamsize size );
    ```

  - 这里 buffer 是一块内存的地址，用来存储或读出数据。参数size 是一个整数值，表示要从缓存（buffer）中读出或写入的字符数。

  - **demo**

    ```cpp
    // reading binary file
        #include <iostream>
        #include <fstream.h>
        
        const char * filename = "test.txt";
        
        int main () {
            char * buffer;
            long size;
            ifstream in (filename, ios::in|ios::binary|ios::ate);
            size = in.tellg();
            
            //从beggin偏移0个字节，说明处于开头
            in.seekg (0, ios::beg);
            buffer = new char [size];
            in.read (buffer, size);
            in.close();
            
            cout << "the complete file is in a buffer";
            
            delete[] buffer;
            return 0;
        }
        //运行结果：
        The complete file is in a buffer
    ```





## stream pointers（get size of file）

- 所有输入/输出流对象(i/o streams objects)都有至少一个流指针：
  - ifstream， 类似istream, 有一个被称为get pointer的指针，指向下一个将被读取的元素
  - ofstream, 类似 ostream, 有一个指针 put pointer ，指向写入下一个元素的位置。
  - fstream, 类似 iostream, 同时继承了get 和 put
- 我们可以通过使用以下成员函数来读出或配置这些指向流中读写位置的流指针：
  - **tellg() 和 tellp()**
    - tellp 用于返回写入位置，tellg 则用于返回读取位置
  - **seekg() 和seekp()**
    -  和C的文件操作方式不同的是，C++ I/O系统管理两个与一个文件相联系的指针。一个是读指针，它说明输入操作在文件中的位置；另一个是写指针，它下次写操作的位置。每次执行输入或输出时，相应的指针自动变化。所以，C++的文件定位分为读位置和写位置的定位，对应的成员函数是seekg()和seekp()。seekg()是设置读位置，seekp是设置写位置。
    - 使用这个原型，流指针被改变为指向从文件开始计算的一个绝对位置。要求传入的参数类型与函数 tellg 和tellp 的返回值类型相同。
      -  seekg ( pos_type position );
      -  seekp ( pos_type position );
    - 使用这个原型可以指定由参数direction决定的一个具体的指针开始计算的一个位移(offset)
      -  seekg ( off_type offset, seekdir direction );
      -  seekp ( off_type offset, seekdir direction );
      

| 第二种原型 |                                |
| ---------- | ------------------------------ |
| ios::beg   | 从流开始位置计算的位移         |
| ios::cur   | 从流指针当前位置开始计算的位移 |
| ios::end   | 从流末尾处开始计算的位移       |

- 流指针 get 和 put 的值对文本文件(text file)和二进制文件(binary file)的计算方法都是不同的，因为文本模式的文件中某些特殊字符可能被修改。由于这个原因，建议对以文本文件模式打开的文件总是使用seekg 和 seekp的第一种原型，而且不要对tellg 或 tellp 的返回值进行修改。

- 对二进制文件，你可以任意使用这些函数，应该不会有任何意外的行为产生。




- **demo**

  ```C++
  // obtaining file size
  #include <iostream.h>
  #include <fstream.h>
  
  const char * filename = "test.txt";
  
  int main () {
      long l,m;
      ifstream in(filename, ios::in|ios::binary);
      l = in.tellg();
      
      //把指针放到文件尾，调用tellg得出文件的大小
      in.seekg (0, ios::end);
      m = in.tellg();
      
      in.close();
      cout << "size of " << filename;
      cout << " is " << (m-l) << " bytes.\n";
      return 0;
  }
  
  //结果:
  size of example.txt is 40 bytes.
      
      
  file1.seekg(1234,ios::cur); //把文件的读指针从当前位置向后移1234个字节 
  file2.seekp(1234,ios::beg); //把文件的写指针从文件开头向后移1234个字节 
  ```

  

## Buffers and Synchronization

- 当我们对文件流进行操作的时候，它们与一个streambuf 类型的缓存(buffer)联系在一起。这个缓存（buffer）实际是一块内存空间，作为流(stream)和物理文件的媒介。例如，对于一个输出流， 每次成员函数put (写一个单个字符)被调用，这个字符不是直接被写入该输出流所对应的物理文件中的，而是首先被插入到该流的缓存（buffer）中。

- 当缓存被排放出来(flush)时，它里面的所有数据或者被写入物理媒质中（如果是一个输出流的话），或者简单的被抹掉(如果是一个输入流的话)。这个过程称为同步(synchronization)，它会在以下任一情况下发生：

  - 当文件被关闭时: 在文件被关闭之前，所有还没有被完全写出或读取的缓存都将被同步。
  - 当缓存buffer 满时:缓存Buffers 有一定的空间限制。当缓存满时，它会被自动同步。
  - 控制符明确指明:当遇到流中某些特定的控制符时，同步会发生。这些控制符包括：flush 和endl。
  - 明确调用函数sync(): 调用成员函数sync() (无参数)可以引发立即同步。这个函数返回一个int 值，等于-1 表示流没有联系的缓存或操作失败。
    

   



## dir _findfirst _findnext

- [参考1](https://blog.csdn.net/qq_23845067/article/details/51915200) [参考2](https://www.cnblogs.com/ranjiewen/p/5960976.html)
- **_finddata_t**

```C++
//io.h
struct _finddata_t
{
    unsigned attrib;
                //_A_ARCH（存档）
                //_A_HIDDEN（隐藏）
                //_A_NORMAL（正常）
                //_A_RDONLY（只读）
                //_A_SUBDIR（文件夹）
                //_A_SYSTEM（系统）
    time_t time_create;
                //创建日期
    time_t time_access;
                //最后访问日期
    time_t time_write;
                //最后修改日期
    _fsize_t size;
                //文件大小
    char name[_MAX_FNAME];
                //文件名， _MAX_FNAME表示文件名最大长度
};

```

- **_findfirst** **_findnext**

```C++
long _findfirst(const char *, struct _finddata_t *);
//第一个参数为文件名，可以用"*.*"来查找所有文件，也可以用"*.cpp"来查找.cpp文件。第二个参数是_finddata_t结构体指针。若查找成功，返回文件句柄，若失败，返回-1。

int _findnext(long, struct _finddata_t *);
//第一个参数为文件句柄，第二个参数同样为_finddata_t结构体指针。若查找成功，返回0，失败返回-1。

int _findclose(long);
//只有一个参数，文件句柄。若关闭成功返回0，失败返回-1。
```



- **demo1**

```C++
#include <iostream>
#include <string>
#include <io.h>
using namespace std;
void dir(string path)
{
    long hFile = 0;
    struct _finddata_t fileInfo;
    string pathName, exdName;
    // \\* 代表要遍历所有的类型,如改成\\*.jpg表示遍历jpg类型文件
    if ((hFile = _findfirst(pathName.assign(path).append("\\*").c_str(), &fileInfo)) == -1) {
        return;
    }
    do
    {
        //判断文件的属性是文件夹还是文件
       cout << fileInfo.name << (fileInfo.attrib&_A_SUBDIR ? "[folder]" : "[file]") << endl;
    } while (_findnext(hFile, &fileInfo) == 0);
    _findclose(hFile);
    return;
}
int main()
{
                //要遍历的目录
    string path = "E:\\intel_tuyoji_pic\\群贤";
    dir(path);
    system("pause");
    return 0;
}
```



- **demo2**

```C++
void dfsFolder(string folderPath, ofstream &fout)
{
    _finddata_t FileInfo;
    string strfind = folderPath + "\\*";
    long Handle = _findfirst(strfind.c_str(), &FileInfo);

    if (Handle == -1L)
    {
        cerr << "can not match the folder path" << endl;
        exit(-1);
    }
    do{
        //判断是否有子目录  
        if (FileInfo.attrib & _A_SUBDIR)
        {
            //这个语句很重要  
            if ((strcmp(FileInfo.name, ".") != 0) && (strcmp(FileInfo.name, "..") != 0))
            {
                string newPath = folderPath + "\\" + FileInfo.name;
                dfsFolder(newPath, fout);
            }
        }
        else
        {
            fout<<folderPath.c_str() << "\\" << FileInfo.name << " ";
            cout << folderPath.c_str() << "\\" << FileInfo.name << endl;
        }
    } while (_findnext(Handle, &FileInfo) == 0);

    _findclose(Handle);
    fout.close();
}
```


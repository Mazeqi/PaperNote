[TOC]

# Web

- [参考](https://zhuanlan.zhihu.com/p/119085959)



## WSAStartup

- [参考](https://blog.csdn.net/richerg85/article/details/7395550?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase)

```C++
  int WSAStartup(
         __in      WORD wVersionRequested,
         __out      LPWSADATA lpWSAData
       ）;
```

- 使用Socket之前必须调用WSAStartup函数，此函数在应用程序中用来初始化Windows Sockets DLL，只有此函数调用成功后，应用程序才可以调用Windows SocketsDLL中的其他API函数，否则后面的任何函数都将调用失败。

- 在WSAStartup函数的入口，Windows Sockets DLL检查了应用程序所需的版本.如果版本高于DLL支持的最低版本,则调用成功并且DLL在wHighVersion中返回它所支持的最高版本，在wVersion中返回它的高版本和wVersionRequested中的较小者，然后Windows Sockets DLL就会假设应用程序将使用wVersion。如果WSDATA结构中的wVersion域对调用方来说不可接收， 它就应调用WSACleanup函数并且要么去另一个Windows Sockets DLL中搜索，要么初始化失败。

- demo

  ```C++
  
  #defineWIN32_LEAN_AND_MEAN
   
  #include<windows.h>
  #include<winsock2.h>
  #include<ws2tcpip.h>
  #include<stdio.h>
   
  //链接  Ws2_32.lib
  #pragmacomment(lib, "ws2_32.lib")
   
   
  int __cdeclmain()
  {
   
      WORD wVersionRequested;
      WSADATA wsaData;
      int err;
   
  /* 使用Windef.h中的 MAKEWORD(lowbyte, highbyte) 宏定义 */
      wVersionRequested = MAKEWORD(2, 2);
   
      err = WSAStartup(wVersionRequested,&wsaData);
      if (err != 0) {
          /* 找不到Winsock DL L.*/
          printf("WSAStartup failed witherror: %d\n", err);
          return 1;
      }
   
  /*确保 WinSock DLL 支持 2.2.*/
  /* Note that ifthe DLL supports versions greater    */
  /* than 2.2 inaddition to 2.2, it will still return */
  /* 2.2 inwVersion since that is the version we     */
  /*requested.                                        */
   
      if (LOBYTE(wsaData.wVersion) != 2 ||HIBYTE(wsaData.wVersion) != 2) {
          /* Tell the user that we could not finda usable */
          /* WinSock DLL.                                  */
          printf("Could not find a usableversion of Winsock.dll\n");
          WSACleanup();
          return 1;
      }
      else
          printf("The Winsock 2.2 dll wasfound okay\n");
         
   
  /* The WinsockDLL is acceptable. Proceed to use it. */
   
  /* Add networkprogramming using Winsock here */
   
  /* then callWSACleanup when done using the Winsock dll */
     
      WSACleanup();
   
  }
  ```

  

## server

- 在socket编程中，服务端和客户端是靠**socket**进行连接的。服务端在建立连接之前需要做的有：
  - 创建socket（伪代码中简称为`socket()`）
  - 将socket与指定的IP和端口（以下简称为port）绑定（伪代码中简称为`bind()`）
  - 让socket在绑定的端口处监听请求（等待客户端连接到服务端绑定的端口）（伪代码中简称为`listen()`）
- ​	而客户端发送连接请求并成功连接之后（这个步骤在伪代码中简称为`accept()`），服务端便会得到**客户端的套接字**，于是所有的收发数据便可以在这个客户端的套接字上进行了。
  - 接收数据：使用客户端套接字拿到客户端发来的数据，并将其存于buff中。（伪代码中简称为`recv()`）
  - 发送数据：使用客户端套接字，将buff中的数据发回去。（伪代码中简称为`send()`）

- 在收发数据之后，就需要断开与客户端之间的连接。在socket编程中，只需要关闭客户端的套接字即可断开连接。（伪代码中简称为`close()`）

```C++
sockfd = socket();    // 创建一个socket，赋给sockfd
bind(sockfd, ip::port和一些配置);    // 让socket绑定端口，同时配置连接类型之类的
listen(sockfd);        // 让socket监听之前绑定的端口
while(true)
{
    connfd = accept(sockfd);    // 等待客户端连接，直到连接成功，之后将客户端的套接字返回出来
    recv(connfd, buff); // 接收到从客户端发来的数据，并放入buff中
    send(connfd, buff); // 将buff的数据发回客户端
    close(connfd);      // 与客户端断开连接
}
```



## client

- 创建socket
- 使用socket和已知的服务端的ip和port连接服务端
- 收发数据
- 关闭连接

```C++
sockfd = socket();    // 创建一个socket，赋给sockfd
connect(sockfd, ip::port和一些配置);    // 使用socket向指定的ip和port发起连接
scanf("%s", buff);    // 读取用户输入
send(sockfd, buff);    // 发送数据到服务端
recv(sockfd, buff);    // 从服务端接收数据
close(sockfd);        // 与服务器断开连接
```



## socket函数

- socket函数用于创建套接字，更严谨的讲是创建一个**套接字描述符**
- 套接字描述符本质上类似于文件描述符，文件通过文件描述符供程序进行读写，而套接字描述符本质上也是提供给程序可以对其缓存区进行读写，程序在其写缓存区写入数据，写缓存区的数据通过网络通信发送至另一端的相同套接字的读缓存区，另一端的程序使用相同的套接字在其读缓存区上读取数据，这样便完成了一次网络数据传输。
- socket函数的参数便是用于设置这个套接字描述符的属性

```c++
int socket(int family, int type, int protocol);
```



- **family parameter**
  - 该参数指明要创建的sockfd的协议族，一般比较常用的有两个
  - **AF_INET**:IPV4
  - **AF_INET6**:IPV6
- **type parameter**
  - 该参数用于指明套接字类型
  - **SOCK_STREAM**:字节流套接字，适用于TCP或SCTP协议
  - **SOCK_DGRAM**：数据报套接字，使用于UDP协议
  - **SOCK_SEQPACKET**:有序分组套接字，适用于SCTP协议
  - **SOCK_RAW**:原始套接字，适用于绕过传输层直接与网络层协议（IPv4/IPv6）通信
- **protocal parameter**
  - 该参数用于指定协议类型。
  - 如果是TCP协议的话就填写`IPPROTO_TCP`，UDP和SCTP协议类似。
  - 也可以直接填写0，这样的话则会默认使用`family`参数和`type`参数组合制定的默认协议
- **return**
  - `socket`函数在成功时会返回套接字描述符，失败则返回-1。
  - 失败的时候可以通过输出`errno`来详细查看具体错误类型。
- **about errno**
  - 通常一个内核函数运行出错的时候，它会定义全局变量`errno`并赋值
  - 当我们引入`errno.h`头文件时便可以使用这个变量。并利用这个变量查看具体出错原因。
  - 一共有两种查看的方法：
    - 直接输出`errno`，根据输出的错误码进行Google搜索解决方案
    - 借助`strerror()`函数，使用`strerror(errno)`得到一个具体描述其错误的字符串。一般可以通过其描述定位问题所在，实在不行也可以拿这个输出去Google搜索解决方案



## bind

- bind函数用于将套接字与一个`ip::port`绑定。或者更应该说是**把一个本地协议地址赋予一个套接字**
- 三个参数：第一个是套接字描述符，第二个是**套接字地址结构体**，第三个是套接字地址结构体的长度。其含义就是将第二个的套接字地址结构体赋给第一个的套接字描述符所指的套接字。

```C++
int bind(int sockfd, const struct sockaddr *myaddr, socklen_t addrlen);
```



### 套接字地址结构体

- 在bind函数的参数表中出现了一个名为`sockaddr`的结构体，这个便是用于存储将要赋给套接字的地址结构的**通用套接字地址结构**。其定义如下：

  ```C++
  #include<windows.networking.sockets.h>
  struct sockaddr
  {
      uint8_t     sa_len;
      sa_family_t sa_family;      // 地址协议族
      char        sa_data[14];    // 地址数据
  };
  ```

- 我们一般不会直接使用这个结构来定义套接字地址结构体，而是使用更加特定化的**IPv4套接字地址结构体**或**IPv6套接字地址结构体**。

- IPv4套接字地址结构体的定义如下：

  ```C++
  #include <netinet/in.h>
  struct in_addr
  {
      in_addr_t       s_addr;         // 32位IPv4地址
  };
  struct sockaddr_in
  {
      uint8_t         sin_len;        // 结构长度，非必需
      sa_family_t     sin_family;     // 地址族，一般为AF_****格式，常用的是AF_INET
      in_port_t       sin_port;       // 16位TCP或UDP端口号
      struct in_addr  sin_addr;       // 32位IPv4地址
      char            sin_zero[8];    // 保留数据段，一般置零
  };
  
  //vs2019
  struct sockaddr_in {
          short   sin_family;
          u_short sin_port;
          struct  in_addr sin_addr;
          char    sin_zero[8];
  };
  
  typedef struct in_addr {
          union {
                  struct { UCHAR s_b1,s_b2,s_b3,s_b4; } S_un_b;
                  struct { USHORT s_w1,s_w2; } S_un_w;
                  ULONG S_addr;
          } S_un;
  #define s_addr  S_un.S_addr /* can be used for most tcp & ip code */
  #define s_host  S_un.S_un_b.s_b2    // host on imp
  #define s_net   S_un.S_un_b.s_b1    // network
  #define s_imp   S_un.S_un_w.s_w2    // imp
  #define s_impno S_un.S_un_b.s_b4    // imp #
  #define s_lh    S_un.S_un_b.s_b3    // logical host
  } IN_ADDR, *PIN_ADDR, FAR *LPIN_ADDR;
  ```

  

- **demo**

  ```C++
  #define DEFAULT_PORT 16555
  // ...
  struct sockaddr_in servaddr;    // 定义一个IPv4套接字地址结构体
  // ...
  bzero(&servaddr, sizeof(servaddr));    // 将该结构体的所有数据置零
  servaddr.sin_family = AF_INET;    // 指定其协议族为IPv4协议族
  servaddr.sin_addr.s_addr = htonl(INADDR_ANY);    // 指定IP地址为通配地址
  servaddr.sin_port = htons(DEFAULT_PORT);    // 指定端口号为16555
  // 调用bind，注意第二个参数使用了类型转换，第三个参数直接取其sizeof即可
  if (-1 == bind(sockfd, (struct sockaddr*)&servaddr, sizeof(servaddr)))
  {
      printf("Bind error(%d): %s\n", errno, strerror(errno));
      return -1;
  }
  ```

- 在指定IP地址的时候，一般就是使用像上面那样的方法指定为通配地址，此时就交由内核选择IP地址绑定。指定特定IP的操作在讲connect函数的时候会提到。

- 在指定端口的时候，可以直接指定端口号为0，此时表示端口号交由内核选择（也就是进程不指定端口号）。但一般而言对于服务器来说，不指定端口号的情况是很罕见的，因为服务器一般都需要暴露一个端口用于让客户端知道并作为连接的参数。

- 注意到不管是赋值IP还是端口，都不是直接赋值，而是使用了类似`htons()`或`htonl()`的函数，这便是**字节排序函数**。

  

### 字节排序函数

- 不同的机子上对于多字节变量的字节存储顺序是不同的，有**大端字节序**和**小端字节序**两种。

- 那这就意味着，将机子A的变量原封不动传到机子B上，其值可能会发生变化（本质上数据没有变化，但如果两个机子的字节序不一样的话，解析出来的值便是不一样的）。这显然是不好的。

- **网络字节序**：

  - 机子A先将变量由自身的字节序转换为网络字节序
  - 发送转换后的数据
  - 机子B接到转换后的数据之后，再将其由网络字节序转换为自己的字节序

- **windows/winsock.h**

  - returns
    - 若成功则返回0，否则返回-1并置相应的`errno`。
    - 比较常见的错误是错误码`EADDRINUSE`（"Address already in use"，地址已使用）。

  ```C++
  u_long PASCAL FAR htonl ( _In_ u_long hostlong);
  
  u_short PASCAL FAR htons (_In_ u_short hostshort);
  
  unsigned long PASCAL FAR inet_addr (_In_z_ const char FAR * cp);
  
  char FAR * PASCAL FAR inet_ntoa (_In_ struct in_addr in);
  
  int PASCAL FAR listen (
                         _In_ SOCKET s,
                         _In_ int backlog);
  
  u_long PASCAL FAR ntohl (_In_ u_long netlong);
  
  u_short PASCAL FAR ntohs (_In_ u_short netshort);
  ```

  

## listen

- listen函数的作用就是开启套接字的监听状态，也就是将套接字从`CLOSE`状态转换为`LISTEN`状态。

```C++
int listen(int sockfd, int backlog);
```

- 其中，`sockfd`为要设置的套接字，`backlog`为服务器处于`LISTEN`状态下维护的队列长度和的最大值。
- **backlog**
  - 这是一个**可调参数**。
  - 服务器套接字处于`LISTEN`状态下所维护的**未完成连接队列（SYN队列）**和**已完成连接队列(Accept队列)**的长度和的最大值。这个是原本的意义，现在的`backlog`仅指**Accept队列的最大长度**，SYN队列的最大长度由系统的另一个变量决定。
  - 这两个队列用于维护与客户端的连接，其中：
    - 客户端发送的SYN到达服务器之后，服务端返回SYN/ACK，并将该客户端放置SYN队列中（第一次+第二次握手）
    - 当服务端接收到客户端的ACK之后，完成握手，服务端将对应的连接从SYN队列中取出，放入Accept队列，等待服务器中的accept接收并处理其请求（第三次握手）
- **backlog 调参**
  - `backlog`是由程序员决定的，不过最后的队列长度其实是`min(backlog, /proc/sys/net/core/somaxconn , net.ipv4.tcp_max_syn_backlog )`，后者直接读取对应位置文件就有了。
  - 事实上`backlog`仅仅是与**Accept队列的最大长度**相关的参数，实际的队列最大长度视不同的操作系统而定。例如说MacOS上使用传统的Berkeley算法基于`backlog`参数进行计算，而Linux2.4.7上则是直接等于`backlog+3`。
  - returns
    - 若成功则返回0，否则返回-1并置相应的`errno`。



## connect

- 该函数用于客户端跟绑定了指定的ip和port并且处于`LISTEN`状态的服务端进行连接。在调用connect函数的时候，调用方（也就是客户端）便会主动发起TCP三次握手。

```C++
int connect(int sockfd, const struct sockaddr *myaddr, socklen_t addrlen);
```

- 其中第一个参数为客户端套接字，第二个参数为用于指定服务端的ip和port的套接字地址结构体，第三个参数为该结构体的长度。

- 操作上比较类似于服务端使用bind函数（虽然做的事情完全不一样），唯一的区别在于指定ip这块。服务端调用bind函数的时候无需指定ip，但客户端调用connect函数的时候则需要指定服务端的ip。

- 在客户端的代码中，令套接字地址结构体指定ip的代码如下(这个就涉及到**ip地址的表达格式与数值格式相互转换**的函数)：

   ```C++
  inet_pton(AF_INET, SERVER_IP, &servaddr.sin_addr);
  ```

- **IP地址格式转换函数**(IP地址一共有两种格式)

   - 表达格式：也就是我们能看得懂的格式，例如`"192.168.19.12"`这样的字符串
   - 数值格式：可以存入套接字地址结构体的格式，数据类型为整型
   
    ```C++
     int inet_pton(int family, const char *strptr, void *addrptr);
     const char *inet_ntop(int family, const void *addrptr, char *strptr, size_t len);
    ```
   
   - `inet_ntop()`函数用于将IP地址从数值格式转换为表达格式
     -  第一个参数指定协议族
   
     -  第二个参数指定要转换的数值格式的IP地址
   
     -  第三个参数指定用于存储转换结果的指针
   
     -  第四个参数指定第三个参数指向的空间的大小，用于防止缓存区溢出,第四个参数可以使用预设的变量：
   
       ```C++
       #define INET_ADDRSTRLEN    16  // IPv4地址的表达格式的长度
       #define INET6_ADDRSTRLEN 46    // IPv6地址的表达格式的长度
       ```
   
     - returns
   
       - 若转换成功则返回指向返回结果的指针
       - 若出错则返回NULL
   
- **connect returns**

   - 若成功则返回0，否则返回-1并置相应的`errno`。
   - 其中connect函数会出错的几种情况：

   - 若客户端在发送SYN包之后长时间没有收到响应，则返回`ETIMEOUT`错误

   - - 一般而言，如果长时间没有收到响应，客户端会重发SYN包，若超过一定次数重发仍没响应的话则会返回该错误

     - 可能的原因是目标服务端的IP地址不存在

       

   - 若客户端在发送SYN包之后收到的是RST包的话，则会立刻返回`ECONNREFUSED`错误

   - - 当客户端的SYN包到达目标机之后，但目标机的对应端口并没有正在`LISTEN`的套接字，那么目标机会发一个RST包给客户端

     - 可能的原因是目标服务端没有运行，或者没运行在客户端知道的端口上

       

   - 若客户端在发送SYN包的时候在中间的某一台路由器上发生ICMP错误，则会发生`EHOSTUNREACH`或`ENETUNREACH`错误

   - - 事实上跟处理未响应一样，为了排除偶然因素，客户端遇到这个问题的时候会保存内核信息，隔一段时间之后再重发SYN包，在多次发送失败之后才会报错
     - 路由器发生ICMP错误的原因是，路由器上根据目标IP查找转发表但查不到针对目标IP应该如何转发，则会发生ICMP错误
     - 可能的原因是目标服务端的IP地址不可达，或者路由器配置错误，也有可能是因为电波干扰等随机因素导致数据包错误，进而导致路由无法转发

- 由于connect函数在发送SYN包之后就会将自身的套接字从`CLOSED`状态置为`SYN_SENT`状态，故当connect报错之后需要主动将套接字状态置回`CLOSED`。此时需要通过调用close函数主动关闭套接字实现。

   ```C++
    if (-1 == connect(sockfd, (struct sockaddr*)&servaddr, sizeof(servaddr)))
    {
        printf("Connect error(%d): %s\n", errno, strerror(errno));
        close(sockfd);        // 新增代码，当connect出错时需要关闭套接字
        return -1;
    }
   ```

    

## accpet

- 该函数用于跟客户端建立连接，并返回客户端套接字。
- accept函数由TCP服务器调用，用于从**Accept队列**中pop出一个已完成的连接。若Accept队列为空，则accept函数所在的进程阻塞。

```C++
int accept(int sockfd, struct sockaddr *cliaddr, socklen_t *addrlen);
```

- 其中第一个参数为服务端自身的套接字，第二个参数用于接收客户端的套接字地址结构体，第三个参数用于接收第二个参数的结构体的长度。
- **returns**
  - 当accept函数成功拿到一个已完成连接时，其会返回该连接对应的**客户端套接字描述符**，用于后续的数据传输。
  - 若发生错误则返回-1并置相应的`errno`。



## recv-send

- recv函数用于通过套接字接收数据，send函数用于通过套接字发送数据

```C++

//linux
ssize_t recv(int sockfd, void *buff, size_t nbytes, int flags);
ssize_t send(int sockfd, const void *buff, size_t nbytes, int flags);

//windows
int PASCAL FAR recv (
        _In_ SOCKET s,
        _Out_writes_bytes_to_(len, return) __out_data_source(NETWORK) char FAR * buf,
        _In_ int len,
        _In_ int flags
);

int PASCAL FAR send (
   		_In_ SOCKET s,
    	_In_reads_bytes_(len) const char FAR * buf,
    	_In_ int len,
    	_In_ int flags
);
```

- Args

  - 第一个参数为要读写的套接字
  -  第二个参数指定要接收数据的空间的指针（recv）或要发送的数据（send）
  -  第三个参数指定最大读取的字节数（recv）或发送的数据的大小（send）
  -  第四个参数用于设置一些参数，默认为0
  -  事实上，去掉第四个参数的情况下，recv跟read函数类似，send跟write函数类似。这两个函数的本质也是一种通过描述符进行的IO，只是在这里的描述符为套接字描述符。

- returns

  - 在recv函数中：

    - 若成功，则返回所读取到的字节数
    - 否则返回-1，置`errno`

    在send函数中：

    - 若成功，则返回成功写入的字节数
    - 事实上，当返回值与`nbytes`不等时，也可以认为其出错。
    - 否则返回-1，置`errno`



## closesocket

- 该函数用于断开连接。或者更具体的讲，该函数用于关闭套接字，并终止TCP连接。

```C++
int closesocket(int sockfd);
```

- returns
  - 同样的，若close成功则返回0，否则返回-1并置`errno`。
  - 常见的错误为**关闭一个无效的套接字**。



## gethostbyname-gethostbyaddr

- [参考1](https://blog.csdn.net/tg5156/article/details/6587871)  [参考2](https://blog.csdn.net/qq_36573828/article/details/81356638)

- gethostbyname函数是通过主机名称获取主机的完整信息。name参数是目标主机的主机 名称。
- gethostbyname函数是通过主机名称获取主机的完整信息。name参数是目标主机的主机 名称。
- 两个函数的返回都是hostent结构体类型指针。hostent结构体定义如下：

```C++
struct hostent{
    //主机名
    char *h_name;
    
    //主机别名列表，可能有多个
    char **h_aliases;
    
    //地址类型
    int h_addrtype;
    
    //地址长度
    int h_length;
    
    //按照网络字节列出的主机ip地址族
    char **h_addr_list;
}

//因为hostent结构支持多种地址类型，所以其定义的h_addr_list是char **型。gethostbyname以后，实际存储情况是这样：
hostent->h_addr_list[0][0] = 127
hostent->h_addr_list[0][1] = 0
hostent->h_addr_list[0][2] = 0
hostent->h_addr_list[0][3] = 1

//demo1
struct hostent* host;
host = gethostbyaddr((char *)&servaddr.sin_addr,sizeof(servaddr.sin_addr)- 1,AF_INET);
char* ip;
ip = inet_ntoa(*(struct in_addr*)*host->h_addr_list);
cout << ip << endl;

//demo2

ret = gethostname(hostname,sizeof(hostname));
printf("the hostname:%s\n",hostname);
hostname[strlen(hostname)+1] = '\0';
/*利用主机名称获取完整的主机信息*/
host = gethostbyname(hostname);
printf("the hostname after getosbyname: %s \n",host->h_name);

```



## inet_pton

- ip地址转化函数

  ```C++
  //SERVER_IP = "127.0.0.1"
  inet_pton(AF_INET, SERVER_IP, &servaddr.sin_addr);
  ```

  

## server demo

```C++
#define _CRT_SECURE_NO_WARNINGS
#include<winsock2.h>
#include<cstdio>
#include<iostream>
#include<cstring>
#include<cstdlib>
#include<signal.h>
#pragma comment(lib, "ws2_32.lib") 

#define DEFAULT_PORT 16555    // 指定端口为16555
#define MAXLINK 2048
#define BUFFSIZE 2048

using namespace std;

int sockfd;
int connfd;

void stopServerRunning(int p)
{
	closesocket(sockfd);
	WSACleanup();
	printf("Close Server\n");
	exit(0);
}

int main() {

	//初始化WSA  
	WORD sockVersion = MAKEWORD(2, 2);
	WSADATA wsaData;
	if (WSAStartup(sockVersion, &wsaData) != 0)
	{
		return 0;
	}


	//用于存放ip和端口的结构
	struct sockaddr_in servaddr;

	//用于收发数据
	char buff[BUFFSIZE];

	sockfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	//printf("%d", sockfd);
	//判断是否出错
	if (-1 == sockfd) {
		printf("Create socket error(%d): %s\n", errno, strerror(errno));
		return -1;
	}


	//将数据仓清零
	memset(&servaddr, 0, sizeof(servaddr));

	//设置ip版本
	servaddr.sin_family = AF_INET;

	//指定ip地址位通配版本
	servaddr.sin_addr.s_addr = htonl(INADDR_ANY);

	//设置端口
	servaddr.sin_port = htons(DEFAULT_PORT);

	//第二个参数进行了参数类型转化，转化为ipv4
	if (-1 == bind(sockfd, (struct sockaddr*)&servaddr, sizeof(servaddr))) {
		printf("Bind error(%d): %s\n", errno, strerror(errno));
		return -1;
	}

	if (-1 == listen(sockfd, MAXLINK))
	{
		printf("Listen error(%d): %s\n", errno, strerror(errno));
		return -1;
	}

	printf("Listening...\n");

	while (true) {
		// 这句用于在输入Ctrl+C的时候关闭服务器
		signal(SIGINT,stopServerRunning);

		connfd = accept(sockfd, NULL, NULL);

		if (-1 == connfd)
		{
			printf("Accept error(%d): %s\n", errno, strerror(errno));
			return -1;
		}

		memset(buff, 0, BUFFSIZE);

		recv(connfd,buff,BUFFSIZE - 1, 0);
		
		printf("Recv: %s\n", buff);
		
		send(connfd, buff, strlen(buff), 0);
	
		closesocket(connfd);

	}
	
	//system("pause");
	return 0;
}
```



## client demo

```C++
#define _CRT_SECURE_NO_WARNINGS
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include<cstdio>
#include<WinSock2.h>
#include <Ws2tcpip.h>
#include<iostream>
#pragma comment(lib,"Ws2_32.lib")

#define BUFFSIZE 2048
#define SERVER_IP "127.0.0.1"
#define SERVER_PORT 16555

using namespace std;
int main() {

	//初始化WSA  
	WORD sockVersion = MAKEWORD(2, 2);
	WSADATA wsaData;
	if (WSAStartup(sockVersion, &wsaData) != 0)
	{
		return 0;
	}

	struct sockaddr_in servaddr;

	char buff[BUFFSIZE];

	int sockfd;

	sockfd = socket(AF_INET, SOCK_STREAM, 0);

	if (-1 == sockfd) {

		printf("Create socket error(%d): %S\n", errno, strerror(errno));
		return -1;

	}

	memset(&servaddr, 0, sizeof(servaddr));

	servaddr.sin_family = AF_INET;
	
	//ip地址转化函数
	inet_pton(AF_INET, SERVER_IP, &servaddr.sin_addr);

	servaddr.sin_port = htons(SERVER_PORT);

	if (-1 == connect(sockfd, (struct sockaddr*)&servaddr, sizeof(servaddr))) {
		printf("Connect error(%d):%s\n", errno,strerror(errno));
		return -1;
	}

	struct hostent* host;

	host = gethostbyaddr((char *)&servaddr.sin_addr,sizeof(servaddr.sin_addr)- 1,AF_INET);

	char* ip;

	ip = inet_ntoa(*(struct in_addr*)*host->h_addr_list);

	cout << ip << endl;
	

	printf("please input ： ");

	scanf("%s", buff);

	send(sockfd, buff, strlen(buff), 0);

	memset(buff, 0, sizeof(buff));

	recv(sockfd, buff, BUFFSIZE - 1, 0);

	printf("Recv: %s\n",buff);
	
	closesocket(sockfd);

	WSACleanup();
	
	return 0;
}
```


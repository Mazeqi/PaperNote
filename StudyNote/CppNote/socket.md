[TOC]

# Web

- [参考](https://zhuanlan.zhihu.com/p/119085959)

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




[TOC]

# Thread

- [参考1]([https://blog.csdn.net/weixin_36049506/article/details/93333549?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522159273325019724843310906%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=159273325019724843310906&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-93333549.first_rank_ecpm_v3_pc_rank_v2&utm_term=C%2B%2BThread](https://blog.csdn.net/weixin_36049506/article/details/93333549?ops_request_misc=%7B%22request%5Fid%22%3A%22159273325019724843310906%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=159273325019724843310906&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-93333549.first_rank_ecpm_v3_pc_rank_v2&utm_term=C%2B%2BThread))
- [参考2](https://blog.csdn.net/Kprogram/article/details/89472995)



# std::move

- [参考1](https://www.jianshu.com/p/b90d1091a4ff) [参考2](https://zhuanlan.zhihu.com/p/94588204)

move函数的参数T&&是一个指向模板类型参数的右值引用【规则2】，通过引用折叠，此参数可以和任何类型的实参匹配，因此move既可以传递一个左值，也可以传递一个右值；

std::move(string("hello"))调用解析：

- 首先，根据模板推断规则，确地T的类型为string;
- typename remove_reference<T>::type && 的结果为 string &&;
- move函数的参数类型为string&&;
- static_cast<string &&>(t)，t已经是string&&，于是类型转换什么都不做，返回string &&;

string s1("hello"); std::move(s1); 调用解析：

- 首先，根据模板推断规则，确定T的类型为string&;
- typename remove_reference<T>::type && 的结果为 string&
- move函数的参数类型为string& &&，引用折叠之后为string&;
- static_cast<string &&>(t)，t是string&，经过static_cast之后转换为string&&, 返回string &&;

```C++
template<typename T>
typename remove_reference<T>::type && move(T&& t)
{
    return static_cast<typename remove_reference<T>::type &&>(t);
}
```



# 线程安全的栈

```C++
#include <exception>
#include <memory>
#include <mutex>
#include <stack>

struct empty_stack: std::exception
{
  const char* what() const throw() {
    return "empty stack!";
  };
};

template<typename T>
class threadsafe_stack
{
private:
  std::stack<T> data;
  mutable std::mutex m;

public:
  threadsafe_stack()
    : data(std::stack<T>()){}

  threadsafe_stack(const threadsafe_stack& other)
  {
    std::lock_guard<std::mutex> lock(other.m);
    data = other.data; // 1 在构造函数体中的执行拷贝
  }

  threadsafe_stack& operator=(const threadsafe_stack&) = delete;

  void push(T new_value)
  {
    std::lock_guard<std::mutex> lock(m);
    data.push(new_value);
  }

  std::shared_ptr<T> pop()
  {
    std::lock_guard<std::mutex> lock(m);
    if(data.empty()) throw empty_stack(); // 在调用pop前，检查栈是否为空

    std::shared_ptr<T> const res(std::make_shared<T>(data.top())); // 在修改堆栈前，分配出返回值
    data.pop();
    return res;
  }

  void pop(T& value)
  {
    std::lock_guard<std::mutex> lock(m);
    if(data.empty()) throw empty_stack();

    value=data.top();
    data.pop();
  }

  bool empty() const
  {
    std::lock_guard<std::mutex> lock(m);
    return data.empty();
  }
};
```



# 向线程传递引用

- [参考](http://shouce.jb51.net/cpp_concurrency_in_action/content/chapter2/2.2-chinese.html)

```c++
void update_data_for_widget(widget_id w,widget_data& data); // 1
void oops_again(widget_id w)
{
  widget_data data;
  std::thread t(update_data_for_widget,w,data); // 2
  display_status();
  t.join();
  process_widget_data(data); // 3
}

std::thread t(update_data_for_widget,w,std::ref(data));
//update_data_for_widget就会接收到一个data变量的引用，而非一个data变量拷贝的引用。
```



# 向线程以及成员函数传递参数

- [参考](http://shouce.jb51.net/cpp_concurrency_in_action/content/chapter2/2.2-chinese.html)

- 这段代码中，新线程将my_x.do_lengthy_work()作为线程函数；my_x的地址①作为指针对象提供给函数。也可以为成员函数提供参数：`std::thread`构造函数的第三个参数就是成员函数的第一个参数，以此类推(代码如下，译者自加)。

```C++
class X
{
public:
  void do_lengthy_work();
};
X my_x;
std::thread t(&X::do_lengthy_work,&my_x); // 1


class X
{
public:
  void do_lengthy_work(int);
};
X my_x;
int num(0);
std::thread t(&X::do_lengthy_work, &my_x, num);
```



# 层级互斥

```C++

class hierarchical_mutex
{
  std::mutex internal_mutex;

  unsigned long const hierarchy_value;
  unsigned long previous_hierarchy_value;

  static thread_local unsigned long this_thread_hierarchy_value;  // 1

  void check_for_hierarchy_violation()
  {
    if(this_thread_hierarchy_value <= hierarchy_value)  // 2
    {
      throw std::logic_error(“mutex hierarchy violated”);
    }
  }

  void update_hierarchy_value()
  {
    previous_hierarchy_value=this_thread_hierarchy_value;  // 3
    this_thread_hierarchy_value=hierarchy_value;
  }

public:
  explicit hierarchical_mutex(unsigned long value):
      hierarchy_value(value),
      previous_hierarchy_value(0)
  {}

  void lock()
  {
    check_for_hierarchy_violation();
    internal_mutex.lock();  // 4
    update_hierarchy_value();  // 5
  }

  void unlock()
  {
    this_thread_hierarchy_value=previous_hierarchy_value;  // 6
    internal_mutex.unlock();
  }

  bool try_lock()
  {
    check_for_hierarchy_violation();
    if(!internal_mutex.try_lock())  // 7
      return false;
    update_hierarchy_value();
    return true;
  }
};
thread_local unsigned long
     hierarchical_mutex::this_thread_hierarchy_value(ULONG_MAX);  // 8
```



# 线程安全的对列

```C++
#include <queue>
#include <memory>
#include <mutex>
#include <condition_variable>

template<typename T>
class threadsafe_queue
{
private:
  mutable std::mutex mut;  // 1 互斥量必须是可变的 
  std::queue<T> data_queue;
  std::condition_variable data_cond;
public:
  threadsafe_queue()
  {}
  threadsafe_queue(threadsafe_queue const& other)
  {
    std::lock_guard<std::mutex> lk(other.mut);
    data_queue=other.data_queue;
  }

  void push(T new_value)
  {
    std::lock_guard<std::mutex> lk(mut);
    data_queue.push(new_value);
    data_cond.notify_one();
  }

  void wait_and_pop(T& value)
  {
    std::unique_lock<std::mutex> lk(mut);
    data_cond.wait(lk,[this]{return !data_queue.empty();});
    value=data_queue.front();
    data_queue.pop();
  }

  std::shared_ptr<T> wait_and_pop()
  {
    std::unique_lock<std::mutex> lk(mut);
    data_cond.wait(lk,[this]{return !data_queue.empty();});
    std::shared_ptr<T> res(std::make_shared<T>(data_queue.front()));
    data_queue.pop();
    return res;
  }

  bool try_pop(T& value)
  {
    std::lock_guard<std::mutex> lk(mut);
    if(data_queue.empty())
      return false;
    value=data_queue.front();
    data_queue.pop();
    return true;
  }

  std::shared_ptr<T> try_pop()
  {
    std::lock_guard<std::mutex> lk(mut);
    if(data_queue.empty())
      return std::shared_ptr<T>();
    std::shared_ptr<T> res(std::make_shared<T>(data_queue.front()));
    data_queue.pop();
    return res;
  }

  bool empty() const
  {
    std::lock_guard<std::mutex> lk(mut);
    return data_queue.empty();
  }
};
```



# 多线程快排

```C++
#include <iostream>
#include <thread>  //①
#include<conio.h>
#include<list>
#include <functional>
#include <algorithm> 
#include<future>
using namespace std;

template<typename T>
std::list<T> parallel_quick_sort(std::list<T> input);

int main() {


    list <int> t;
    t.push_back(2);
    t.push_back(3);
    t.push_back(1);
    t.push_back(5);
    t.push_back(123);
    t.push_back(222);
    t.push_back(22);
    t.push_back(20);
    t.push_back(55);


    list<int> result = parallel_quick_sort(t);
    for (list<int>::iterator ite = result.begin(); ite != result.end(); ite++) {
        std::cout << *ite << endl;
    }
    return 0;
}


template<typename T>
std::list<T> parallel_quick_sort(std::list<T> input)
{
    if (input.empty())
    {
        return input;
    }
    std::list<T> result;
    result.splice(result.begin(), input, input.begin());
    T const& pivot = *result.begin();

    auto divide_point = std::partition(input.begin(), input.end(),
        [&](T const& t) {return t < pivot; });

    std::list<T> lower_part;
    lower_part.splice(lower_part.end(), input, input.begin(),
        divide_point);

    std::future<std::list<T> > new_lower(  // 1
        std::async(&parallel_quick_sort<T>, std::move(lower_part)));

    auto new_higher(
        parallel_quick_sort(std::move(input)));  // 2

    result.splice(result.end(), new_higher);  // 3
    result.splice(result.begin(), new_lower.get());  // 4
    return result;
}
```


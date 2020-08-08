# os

## walk

- [参考](https://www.jianshu.com/p/bbad16822eab)

- top 是你所要便利的目录的地址

- topdown 为真，则优先遍历top目录，否则优先遍历top的子目录(默认为开启)

- onerror 需要一个 callable 对象，当walk需要异常时，会调用

- followlinks 如果为真，则会遍历目录下的快捷方式(linux 下是 symbolic link)实际所指的目录(默认关闭)

```python
# 每次遍历的对象都是返回的是一个三元组(root,dirs,files)
from os import walk
walk(top, topdown=True, onerror=None, followlinks=False)

import os

Root = 'a'
Dest = 'b'

for (root, dirs, files) in os.walk(Root):
    new_root = root.replace(Root, Dest, 1)
    if not os.path.exists(new_root):
        os.mkdir(new_root)
    
    for d in dirs:
        d = os.path.join(new_root, d)
        if not os.path.exists(d):
            os.mkdir(d)
    
    for f in files:
        # 把文件名分解为 文件名.扩展名
        # 在这里可以添加一个 filter，过滤掉不想复制的文件类型，或者文件名
        (shotname, extension) = os.path.splitext(f)
        # 原文件的路径
        old_path = os.path.join(root, f)
        new_name = shotname + '_bak' + extension
        # 新文件的路径
        new_path = os.path.join(new_root, new_name)
        try:
            # 复制文件
            open(new_path, 'wb').write(open(old_path, 'rb').read())
        except IOError as e:
            print(e)
```


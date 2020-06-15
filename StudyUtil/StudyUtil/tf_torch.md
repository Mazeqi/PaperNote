[TOC]

# tensorflow

## image

- tensorflow的输入图片的形状是 [batch_size, image_size, image_size, 3]

```python
images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images')
```



## tf.Session

- [参考1](https://blog.csdn.net/u012436149/article/details/52908692)
- [参考2](https://zhuanlan.zhihu.com/p/32869210)

```python
# 我们在编写代码的时候，总是要先定义好整个图，然后才调用sess.run()。那么调用sess.run()的时候，程序是否执行了整个图

# 三参数

# target 用来控制 session 使用的硬件设备， 如果使用空值，那么这个 session 就只会使用本地的设备，如果使用 grpc:// URL，那么就会使用这台服务器控制的所有设备。

#graph 用来控制该 session 运行哪个计算图，如果为空，那么该 session 就只会使用当前的默认 Graph，如果使用多个计算图，就可以在这里指定。

#config 用来 指定一个 tf.ConfigProto 格式的 session 运行配置，比如说它里面包含的 allow_soft_placement 如果指定为 TRUE，那么 session 就会自动把不适合在 GPU 上运行的 OP 全部放到 CPU 上运行；cluster_def 是分布式运行时候需要指定的配置；gpu_options.allow_growth 设置会使得程序在开始时候逐步的增长 GPU 显存使用量，而不是一开始就最大化的使用所有显存。第一个和第三个配置是经常用到的。


# demo1
import tensorflow as tf
state = tf.Variable(0.0,dtype=tf.float32)
one = tf.constant(1.0,dtype=tf.float32)
new_val = tf.add(state, one)
update = tf.assign(state, new_val) #返回tensor， 值为new_val
update2 = tf.assign(state, 10000)  #没有fetch，便没有执行
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        print sess.run(update)


#demo2 带feed_dict
# feed_dict的作用是给使用placeholder创建出来的tensor赋值。其实，他的作用更加广泛：feed 使用一个 值临时替换一个 op 的输出结果. 你可以提供 feed 数据作为 run() 调用的参数. feed 只在调用它的方法内有效, 方法结束, feed 就会消失.

import tensorflow as tf
y = tf.Variable(1)
b = tf.identity(y)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(b,feed_dict={y:3})) #使用3 替换掉
    #tf.Variable(1)的输出结果，所以打印出来3 
    #feed_dict{y.name:3} 和上面写法等价
    print(sess.run(b))  #由于feed只在调用他的方法范围内有效，所以这个打印的结果是 1

#yolov2
output = self.sess.run(self.yolo.logits, feed_dict = {self.yolo.images: image})
```



## tf.ConfigProto

- [参考](https://zhuanlan.zhihu.com/p/78998468)

```python
sess_config = tf.ConfigProto(device_count = {'GPU': 0})

sess_config = tf.ConfigProto(device_count={'GPU':0, 'CPU':4})

with tf.Session(config=sess_config) as sess: # 基本格式
    gan = Model(sess, FLAGS) # 此行仅用于示例
    
# yolov2
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
config = tf.ConfigProto(gpu_options=tf.GPUOptions())
self.sess = tf.Session(config=config)
```




## tf.constant

```python
# 生成常量
a = tf.constant([[1, 2, 3], [4, 5, 6]])

# 在sess中才能输出
with tf.Session() as sess:
    print(c.eval())
```



## tf.Variable   tf.get_variable

- [参考](https://blog.csdn.net/gg_18826075157/article/details/78368924)

```python
# 1.图变量的初始化方法
# 在TensorFlow的世界里，变量的定义和初始化是分开的，所有关于图变量的赋值和计算都要通过tf.Session的run来进行。想要将所有图变量进行集体初始化时应该使用tf.global_variables_initializer。
x = tf.Variable(3, name='x')
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# 2.两种定义图变量的方法

# tf.Variable
# trainab为true，会把它加入到GraphKeys.TRAINABLE_VARIABLES，才能对它使用Optimizer
tf.Variable.init(initial_value, trainable=True, collections=None, validate_shape=True, name=None)

v = tf.Variable(3, name='v')
v2 = v.assign(5)
sess = tf.InteractiveSession()
sess.run(v.initializer)
# 输出3
sess.run(v)


# tf.get_variable
# tf.get_variable跟tf.Variable都可以用来定义图变量，但是前者的必需参数（即第一个参数）并不是图变量的初始值，而是图变量的名称。
init = tf.constant_initializer([5])
x = tf.get_variable('x', shape=[1], initializer=init)
sess = tf.InteractiveSession()
sess.run(x.initializer)
sess.run(x)



# demo
a = tf.Variable([
     [
                  [1, 5, 5, 2],
                  [9, -6, 2, 8],
                  [-3, 7, -9, 1]
              ],
 
              [
                  [-1, 7, -5, 2],
                  [9, 6, 2, 8],
                  [3, 7, 9, 1]
              ],
            
             [
                  [21, 6, -5, 2],
                  [9, 36, 2, 8],
                  [3, 7, 79, 1]
              ]
])
# 变量必须初始化才能输出
with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(a.eval())
```



## tf.global_variables

```python
# tf.global_variables或者tf.all_variables都是获取程序中的变量
variables = tf.global_variables()
variables[0].name
variables[1].name

#yolov2
self.variable_to_restore = tf.global_variables()
self.saver = tf.train.Saver(self.variable_to_restore)
```



## tf.variable_scope  tf.name_scope

```python
# TensorFlow的命名空间分为两种，tf.variable_scope和tf.name_scope。

#1. tf.variable_scope
-------------------------------------------------------------------

# 当使用tf.get_variable定义变量时，如果出现同名的情况将会引起报错
# 而对于tf.Variable来说，却可以定义“同名”变量

with tf.variable_scope('scope'):
     v1 = tf.get_variable('var', [1])
     v2 = tf.get_variable('var', [1])
     #ValueError: Variable scope/var already exists, disallowed. Did you mean to set reuse=True in VarScope? Originally defined at:
        
     v1 = tf.Variable(1, name='var')
     v2 = tf.Variable(2, name='var')
     # 但是把这些图变量的name属性打印出来，就可以发现它们的名称并不是一样的。
        
# 如果想使用tf.get_variable来定义另一个同名图变量，可以考虑加入新一层scope，比如：
with tf.variable_scope('scope1'):
	v1 = tf.get_variable('var', shape=[1])
    with tf.variable_scope('scope2'):
         v2 = tf.get_variable('var', shape=[1])

# yolov2
with tf.variable_scope('',reuse=tf.AUTO_REUSE):
    main()
        
        
        
#2 tf.name_scope
#-----------------------------------------------------------------------------    

# 当tf.get_variable遇上tf.name_scope，它定义的变量的最终完整名称将不受这个tf.name_scope的影响
 with tf.variable_scope('v_scope'):
 	 with tf.name_scope('n_scope'):
  		 x = tf.Variable([1], name='x')
   	     y = tf.get_variable('x', shape=[1], dtype=tf.int32)
 		 z = x + y
 # x.name, y.name, z.name
 # ('v_scope/n_scope/x:0', 'v_scope/x:0', 'v_scope/n_scope/add:0')
    
    
    
#3 图变量的复用
#------------------------------------------------------------------------------

# 如果我们正在定义一个循环神经网络RNN，想复用上一层的参数以提高模型最终的表现效果
 with tf.variable_scope('scope'):
          v1 = tf.get_variable('var', [1])
          tf.get_variable_scope().reuse_variables()
          v2 = tf.get_variable('var', [1])
# v1.name, v2.name
# ('scope/var:0', 'scope/var:0')

# 或者
with tf.variable_scope('scope'):
     v1 = tf.get_variable('x', [1])
        
with tf.variable_scope('scope', reuse=True):
    v2 = tf.get_variable('x', [1])
#  v1.name, v2.name
#  ('scope/x:0', 'scope/x:0') 



# 4 图变量的种类
# --------------------------------------------------------------------------

# TensorFlow的图变量分为两类：local_variables和global_variables。
# 如果我们想定义一个不需要长期保存的临时图变量，可以向下面这样定义它：
with tf.name_scope("increment"):
	 zero64 = tf.constant(0, dtype=tf.int64)
	 current = tf.Variable(zero64, name="incr", trainable=False, collections=[ops.GraphKeys.LOCAL_VARIABLES])

```



## random number（init variable）

- [参考](https://blog.csdn.net/yjk13703623757/article/details/77075711)

```python
# yolov2
weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')


tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

tf.random_uniform(shape, minval=0.0, maxval=1.0, dtype=tf.float32, seed=None, name=None)

tf.random_shuffle(value, seed=None, name=None)

tf.random_crop(value, size, seed=None, name=None) 

tf.multinomial(logits, num_samples, seed=None, name=None) 

tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)

```



## tf.tile

```python
#在batch_size的纬度乘了batch_size,原来这个纬度的通道是1
        self.offset = tf.tile(self.offset, (self.batch_size, 1, 1, 1))
```



## tf.placeholder

```python
#占位符
self.labels = tf.placeholder(tf.float32, [None, self.cell_size, self.cell_size, self.box_per_cell, self.num_class + 5], name = 'labels')
```



## tf.nn.max_pool

```python
#ksize是kernel的大小，一般是[1,x,x,1]第一个和最后一个纬度一般不做池化
# padding='SAME'使得加入填充后，输出与输入的形状一致
pool = tf.nn.max_pool(inputs, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = name)

```



## tf.nn.conv2d

```python
# filter = [filter_height, filter_width, in_channels, out_channels] [3,3,3,32] filter_height 为kernel的高
tf.nn.conv2d(inputs, filter, strides=[1, 1, 1, 1], padding='SAME', name=name)
```



## tf.layer.conv2d

```python
# filters_num 是通道数 kernel_size是卷积核大小
out=tf.layers.conv2d(x,filters=filters_num,kernel_size=filters_size,strides=stride,padding='VALID',activation=None,use_bias=use_bias,name=name)
```



## tf.nn.batch_normalization

```python
# `output = (x - mean) / (sqrt(var) + epsilon) * gamma + beta`
# y=scale∗(x−mean)/var+offset
# 得到通道数，初始图片是3，卷积后为各种长度
depth = shape[3]

# 缩放，默认是1,归一化
scale = tf.Variable(tf.ones([depth, ], dtype='float32'), name='scale')

# 偏移量，beta
shift = tf.Variable(tf.zeros([depth, ], dtype='float32'), name='shift')

# 均值
mean = tf.Variable(tf.ones([depth, ], dtype='float32'), name='rolling_mean')

# 方差
variance = tf.Variable(tf.ones([depth, ], dtype='float32'), name='rolling_variance')

# 1e-05防止除数为0
conv_bn = tf.nn.batch_normalization(conv, mean, variance, shift, scale, 1e-05)
```



## tf.layer.batch_normalize

```python
out=tf.layers.batch_normalization(out,axis=-1,momentum=0.9,training=False,name=name+'_bn')
```



## tf.shape  tensor.get_shape

```python
# 返回元组，不能放到sess.run()里面
tensor.get_shape()

# 返回一个tensor，想要知道多少，必须放到sess.run()
tf.shape()
```



## tf.stack tf.concat

- [参考](https://zhuanlan.zhihu.com/p/37637446)

```python
# tf.concat是沿某一维度拼接shape相同的张量，拼接生成的新张量维度不会增加。而tf.stack是在新的维度上拼接，拼接后维度加1
ab1 = tf.concat([a,b],axis=0)
ab2 = tf.stack([a,b], axis=0)

#yolov2合并几个loss
loss = tf.concat([coo_loss, con_loss, pro_loss], axis = 4)
```



## tf.clip_by_value

```python
#输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。小于min的让它等于min，大于max的元素的值等于max。
tf.clip_by_value(A, min, max)：
```



## tf.reduce_max tf.reduce_mean...

```python
#得到最大值 reduction_indices在哪个纬度进行
tf.reduce_max(input_tensor, reduction_indices=None, keep_dims=False, name=None)

# 求和
tf.reduce_sum(input_tensor,axis=None,keepdims=None,name=None,reduction_indices=None,keep_dims=None）

# 求平均
tf.reduce_mean(input_tensor,reduction_indices=None,keep_dims=False,name=None)
```



## tf.expend_dims tf.reshape

```python
tf.expand_dims(tensor, dim, name)
# 在一维度上拓展
one_img = tf.expand_dims(one_img, 0)

#reshape也可以达到相同效果，但是有些时候在构建图的过程中，placeholder没有被feed具体的值，这时就会包下面的错误：TypeError: Expected binary or unicode string, got 1
tf.reshape(input, shape=[])
```



## tf.squeeze

```python
# 从tensor中删除所有大小是1的维度
# 给定张量输入，此操作返回相同类型的张量，并删除所有尺寸为1的尺寸。 如果不想删除所有尺寸1尺寸，可以通过指定squeeze_dims来删除特定尺寸1尺寸。
squeeze(input,axis=None,name=None,squeeze_dims=None)
```



## tf.train.exponential_decay

```python
# 调整学习率
tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=True/False)

# yolov2
learn_rate = tf.train.exponential_decay(self.initial_learn_rate, self.global_step, 20000, 0.1, name='learn_rate')
```



## tf.train.AdamOptimizer(with else)

- [参考](https://www.jianshu.com/p/e6e8aa3169ca)

```python
tf.train.AdamOptimizer.__init__(
	learning_rate=0.001, 
	beta1=0.9, 
	beta2=0.999, 
	epsilon=1e-08, 
	use_locking=False, 
	name='Adam'
)

# yolov2
optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.yolo.total_loss, global_step=self.global_step)
```



## tf.control_dependencies

- [参考](https://blog.csdn.net/liuweiyuxiang/article/details/79952493)

```python
# session在运行d、e之前会先运行a、b、c。在with tf.control_dependencies之内的代码块受到顺序控制机制的影响。
with tf.control_dependencies([a, b, c]):
	d = ...
    e = ...

#yolov2
self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.yolo.total_loss, global_step=self.global_step)

self.average_op = tf.train.ExponentialMovingAverage(0.999).apply(tf.trainable_variables())

with tf.control_dependencies([self.optimizer]):
    self.train_op = tf.group(self.average_op)
```



## tf.train.ExponentialMovingAverage

- [参考](https://blog.csdn.net/UESTC_C2_403/article/details/72235334)

```python
# tf.train.ExponentialMovingAverage这个函数用于更新参数，就是采用滑动平均的方法更新参数。这个函数初始化需要提供一个衰减速率（decay），用于控制模型的更新速度。这个函数还会维护一个影子变量（也就是更新参数后的参数值），这个影子变量的初始值就是这个变量的初始值，影子变量值的更新方式如下：
# shadow_variable = decay * shadow_variable + (1-decay) * variable
# shadow_variable是影子变量，variable表示待更新的变量，也就是变量被赋予的值，decay为衰减速率。decay一般设为接近于1的数（0.99,0.999）。decay越大模型越稳定，因为decay越大，参数更新的速度就越慢，趋于稳定。

tf.train.ExponentialMovingAverage(decay, steps)

# yolov2
self.average_op = tf.train.ExponentialMovingAverage(0.999).apply(tf.trainable_variables())
```



## tf.train.Saver

- [参考](https://blog.csdn.net/Jerr__y/article/details/78594494)

```python

# 保存模型
# ------------------------------------------------------------------------------
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Create some variables.
v1 = tf.Variable([1.0, 2.3], name="v1")
v2 = tf.Variable(55.5, name="v2")

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

ckpt_path = './ckpt/test-model.ckpt'
# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
sess.run(init_op)
save_path = saver.save(sess, ckpt_path, global_step=1)
print("Model saved in file: %s" % save_path)


# 恢复模型
# ------------------------------------------------------------------------------

# 导入模型之前，必须重新再定义一遍变量。
# 但是并不需要全部变量都重新进行定义，只定义我们需要的变量就行了。
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Create some variables.
v1 = tf.Variable([11.0, 16.3], name="v1")
v2 = tf.Variable(33.5, name="v2")

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
# Restore variables from disk.
ckpt_path = './ckpt/test-model.ckpt'
saver.restore(sess, ckpt_path + '-'+ str(1))
print("Model restored.")

print sess.run(v1)
print sess.run(v2)
```

## tensorboard (tf.summary)

- [参考](https://blog.csdn.net/hongxue8888/article/details/79753679)

```python
#1
#------------------------------------------------------------------------------
# 用来显示标量信息
# 一般在画loss,accuary时会用到这个函数。
tf.summary.scalar(tags, values, collections=None, name=None)
tf.summary.scalar('mean', mean)

#yolov2
tf.summary.scalar('total_loss', self.total_loss)


#2
#------------------------------------------------------------------------------
# 用来显示直方图信息
#一般用来显示训练过程中变量的分布情况
tf.summary.histogram(tags, values, collections=None, name=None)
tf.summary.histogram('histogram', var)


#3
#-----------------------------------------------------------------------------
# merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。如果没有特殊要求，一般用这一句就可一显示训练时的各种信息了。
tf.summaries.merge_all(key='summaries')



#4
#----------------------------------------------------------------------------
# 指定一个文件用来保存图。
tf.summary.FileWritter(path,sess.graph)


#demo
#------------------------------------------------------------------------------
#生成准确率标量图  
tf.summary.scalar('accuracy',acc)   
merge_summary = tf.summary.merge_all() 

#定义一个写入summary的目标文件，dir为写入文件地址  
train_writer = tf.summary.FileWriter(dir,sess.graph)

# ......(交叉熵、优化器等定义)  
#训练循环  
for step in xrange(training_step):               
    #调用sess.run运行图，生成一步的训练过程数据  
    train_summary = sess.run(merge_summary,feed_dict =  {...})
    train_writer.add_summary(train_summary,step)#调用train_writer的add_summary方法将训练过程以及训练步数保存  

    
#yolov2
#-----------------------------------------------------------------------------
self.summary_op = tf.summary.merge_all()

self.writer = tf.summary.FileWriter(self.output_dir)

self.writer.add_graph(self.sess.graph)

summary_, loss, _ = self.sess.run([self.summary_op,self.yolo.total_loss,self.train_op], feed_dict = feed_dict)

self.writer.add_summary(summary_, step)
```


# torch

## torch.clamp

```python
#将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。
# (n1, n2, 2)
intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0) 
torch.clamp(input, min, max, out=None) → Tensor
```


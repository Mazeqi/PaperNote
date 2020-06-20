[TOC]

# ArcFace Loss



## Introduce

- [参考1](https://zhuanlan.zhihu.com/p/76541084)
- [参考2](https://blog.csdn.net/Wuzebiao2016/article/details/81839452)
- Additive Angular Margin loss(加性角度间隔损失函数)，对特征向量和权重归一化，对$\theta$ 加上角度间隔m，角度间隔比余弦间隔在对角度的影响更加直接。几何上有恒定的线性角度margen。
- ArcFace 中是直接在角度空间 $\theta$ 中最大化分类界限，而CosFace是在余弦空间 $\cos (\theta)$ 中最大化分类界限



## Softmax Loss

- 无论是SphereFace、CosineFace还是ArcFace的损失函数，都是基于传统的softmax loss进行修改得到的，所以想要理解ArcFace，需要对之前的损失函数有一定理解。

$$
L_S = - \frac{1}{m} \sum^{m}_{i=1}  \log{  (\frac{e^{W_{y_i}^T x_i + b_{y_i} }  } { \sum^{n}_{j=1}  e^{ W^{T}_{j} x_i + b_j} }) }
$$

- 这是传统的Softmax，$W_j^T x_i + b_j$ 代表全连接层的输出，在损失$L_S$下降的过程中，则必须提高$W^T_{y_i}x_i+b_{y_i}$ 所占有的比重，从而使得该类别的样本更多地落入岛该类的决策边界之内。

- **这种方式主要考虑样本是否能正确分类，缺乏类内和类间距离的约束。**

- 在[A Discriminative Feature Learning Approach for Deep Face Recognition]这篇文章中，作者使用了一个比LeNet更深的网络结构，用Mnist做了一个小实验来证明Softmax学习到的特征与理想状态下的差距。

- 实验结果表明，传统的Softmax仍存在着很大的类内距离，也就是说，通过对损失函数增加类内距离的约束，能达到比更新现有网络结构更加事半功倍的效果。于是，[A Discriminative Feature Learning Approach for Deep Face Recognition]的作者提出了**Center Loss**，并从不同角度对结果的提升做了论证。

  ![](./img/softmax_loss.jpg)

  

## Center Loss

$$
L_C = \frac{1}{2} \sum^{m}_{i=1} \lVert x_i - c_{y_i} \rVert \\


\Delta c_j =\frac{\sum^m_{i=1} \delta(y_i =j) \cdot (c_j - x_i)}{1 + \sum^m_{i=1} \delta(y_i=j)}
$$



- Center Loss的整体思想是希望一个batch中的每个样本的feature离feature 的中心的距离的平方和要越小越好，也就是类内距离要越小越好。作者提出，最终的损失函数包含softmax loss和center loss，用参数λ来控制二者的比重，如下面公式所示：
  $$
  L = L_S + L_C=  - \frac{1}{m} \sum^{m}_{i=1}  \log{  (\frac{e^{W_{y_i}^T x_i + b_{y_i} }  } { \sum^{n}_{j=1}  e^{ W^{T}_{j} x_i + b_j} }) } \ \ + \ \ \frac{\lambda}{2} \sum^{m}_{i=1} {\lVert x_i - c_{yi} \rVert}^2
  $$
  

- 因而，加入了Softmax Loss对正确类别分类的考虑以及Center Loss对类内距离紧凑的考虑，总的损失函数在分类结果上有很好的表现力。以下是作者继上个实验后使用新的损失函数并调节不同的参数$\lambda$ 得到的实验结果，可以看到，加入了Center Loss后增加了对类内距离的约束，使得同个类直接的样本的类内特征距离变得紧凑。

![](./img/center_loss.jpg)


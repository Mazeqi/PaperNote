[TOC]

# ArcFace Loss



## Introduce

- [参考](https://zhuanlan.zhihu.com/p/76541084)

- Additive Angular Margin loss(加性角度间隔损失函数)，对特征向量和权重归一化，对$\theta$ 加上角度间隔m，角度间隔比余弦间隔在对角度的影响更加直接。几何上有恒定的线性角度margen。

- ArcFace 中是直接在角度空间 $\theta$ 中最大化分类界限，而CosFace是在余弦空间 $\cos (\theta)$ 中最大化分类界限
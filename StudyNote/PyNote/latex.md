[TOC]

# latex

- [常用符号](https://www.jianshu.com/p/d6789115018e)

## 分数

$$
f_c = \frac{1}{2}
$$





## 累加

$$
\sum_{x \in x_c}
$$



## 中括号

- [参考](http://www.cppblog.com/luyulaile/archive/2012/08/28/188512.html)
- demo1

$$
\phi(x) = \begin{cases}
				x \text{,           if   x > 0} \\ 
				0.1x \text{,      otherwise}   \\
				0.2x \text{,      x = 0}
\end{cases}
$$

- demo2
  $$
  \begin{cases}
  
  \ \  \mu  = \frac{1}{m} \sum_{i=1}^{m}f_i, 
  \\ \\
  			
  \    \sigma^{2}  = \frac{1}{m} \sum_{i=1}^{m}(f_i - \mu)^2,&(5) 
  \\ \\
  	 				
    f_{bn} = \gamma \cdot \frac{f - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta
  \end{cases}
  $$
  



- demo3
  $$
  \begin{cases}
  		x, 						  &&&&  if \ \ x\geq 0 \\
  		negative\_slope \times x, &&&&  otherwise
  \end{cases}
  $$
  

## 花体 \mathcal  和对齐

- [对齐参考](https://blog.csdn.net/bendanban/article/details/77336206)

$$
\begin{eqnarray}

&（\mathcal L）& 常用来表示损失函数\\
&（\mathcal D）& 表示样本 \\
&（\mathcal N）& 常用来表示高斯分布\\

\end{eqnarray}
$$

$$
\begin{aligned}

&（\mathcal L）& 常用来表示损失函数\\
&（\mathcal D）& 表示样本 \\
&（\mathcal N）& 常用来表示高斯分布\\

\end{aligned}
$$



## y加帽子 - ||

$$
\lVert \hat{y} \rVert
$$



## 实数集 交 并 补 全集

$$
\mathbb{R}
\\
\mathbb{Z}
\\
\mathbb{N}
\\
\subset 子集
\\
\subseteq 真子集
\\
\supset  包含
\\
\in      属于      
\\
\cap    交     
\\
\cup    并     
\\
\mid      或  
\\
\notin    不属于   

$$




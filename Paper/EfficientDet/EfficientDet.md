[TOC]

# Abstract

- first: propose a weighted bi-directional feature pyramid method(BIFPN)

- second: propose a compound scaling method that uniformly scales the resolution, depth, and width for all backbone,feature network, and box/class prediction networks at the same time.



# Introduction



## Two Challenges

### challenge 1

​	**efficient multi-scale feature fusion**(FPN has been widely used for muti-scale feature fusion)

​	While fusing different input features, **most previous** works simply **sum them up without distinction;** however, since these different input features are at different resolutions, we observe **they usually contribute to the fused output feature unequally.**

​	**propose a simple yet highly effective weighted bi-directional feature pyramid network(BiFPN)** to address this issue.

### challenge 2

​	**model scaling**

​	**previous works** mainly rely on bigger backbone networks or larger input image sizes for higher accuracy.

​	**Paper observe** : **scaling up(按比例缩放) feature network and box/class prediction network** is also critical when taking into account both accuracy and efficiency

​	Propose a compound scaling method for object detectors which jointly scales up the resolution/depth/width for all backbone, feature network,box/class prediction network.



# Related work



## BiFPN

### Problem Formulation

​	Formally，given a list of multi-scale features $\overrightarrow{P}^{in} = (P^{in}_{l_1}, p^{in}_{l_2},...)$, where $P^{in}_{l_i}$ represents the feature at level $l_i$ , our goal is to find a transformation f that can effectively aggregate different features and output a list of new features: $\overrightarrow{P}^{out} = f(\overrightarrow{P}^{in})$ 

![](img/EDet-fpn.png)

​		In Figure2 , FPN takes level 3-7 input features $\overrightarrow{P}^{in} = (P^{in}_3,...,P^{in}_7)$  where **$P^{in}_i$ represents a feature level with resolution of $1/2^i$ of the input images.**

​			For instance, if input resolution is 640*640, the $P^{in}_3$ represents feature level 3 $640/2^3 = 80$ with resolution 80 * 80.The FPN aggregates muti-scale features in a top-down manner.
$$
P^{out}_7 = Conv(P^{in}_7) \\
P^{out}_6 = Conv(P^{in}_6 + Resize(P^{out}_7)) \\
... \\
P^{out}_3 = Conv(P^{in}_3 + Resize(P^{out}_4)) \\
$$
 	In the formulation, Resize is usually a **upsampling or downsampling op** for resolution matching, and Conv is usually a convolutional op for feature processing.



### Cross-Scale Connections

​		Paper propose serveral optimizations for cross-scale connections.

​		First: **remove those nodes that have one input edge**. **If a node has only one input edge with no feature fusion**, then it will have less contribution to feature network that aims at fusing different features.

​		Second, **add an extra edge** from the original input to output node if **they are the same level**, in order to fuse more features without adding much cost.

​	   Third, unlike PANet that only has one top-down and one bottom-up path, we **treat each bidirectional(top-down & bottom-up) path as one feature network layer**, and repeat the same layer multiple times to enable more high-level feature fusion.

### Weighted Feature Fusion

​	since different input features are at different resolutions ,they usually contribute to the output feature unequally.

​	**Add an additional weight for each input, and let the network to learn the importance of each input feature.**


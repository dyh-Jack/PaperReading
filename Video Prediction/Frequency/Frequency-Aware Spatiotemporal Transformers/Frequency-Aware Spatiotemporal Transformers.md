#### Frequency-Aware Spatiotemporal Transformers for Video Inpainting Detection

总体结构：

![image-20220216161402674](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220216161402674.png)

文章的整体结构基本使用ViT的结构，首先先进行Embedidng：

![image-20220216161522808](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220216161522808.png)

其中E和$E_0$分别是对于输入patch在时间和空间维度的变换，再加上位置信息$E_{pos}$

之后利用Transformer的Encoder结构：

![image-20220216161646914](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220216161646914.png)

而在Decoder部分，首先Encoder的输出被重塑为$H/16,W/16,D_0$的特征图，之后利用标准的上采样层来提高分辨率，同时在每一次上采样后将其与得到的频率特征相结合，从而获得更好的效果。

文章中采用DCT来提取频率特征：

![image-20220216163639037](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220216163639037.png)

其中$\tau$代表DCT变换，$\tau^{-1}$代表DCT的反变换，而$f_i$代表不同频率的带通滤波器，利用n个滤波器得到不同的频率特征。

在这里，我们设置n为3，一方面由于需要将频率信号与上采样的图像进行堆叠，通道数设置为3与最终3通道图像相一致；另一方面，便于我们将频率信息分解为普通的高频、中频、低频三个模块。之后不断下采样与上采样得到的图像结合。

而在损失函数的设计上，为了解决在视频补画中出现CE损失无法处理的类不平衡的问题，我们采用焦点损失来减少类不平衡的影响：

![image-20220216164208806](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220216164208806.png)

此外，我们采用平均并集相交（mIoU）作为视频补画检测的评价指标。因此，为了培养预测掩膜与GT之间的更多交集，我们采用IoU得分作为损失函数的一部分：

![image-20220216164419000](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220216164419000.png)

最终得到的损失函数如下：

![image-20220216164445658](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220216164445658.png)

两个损失函数在优化过程中发挥了重要作用，Focal loss帮助网络缓解类失衡，关注硬样本。此外，借据损失直接衡量了评价指标，指导框架越来越准确地预测嵌入区域。
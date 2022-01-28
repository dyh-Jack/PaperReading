#### MotionRNN: A Flexible Model for Video Prediction with Spacetime-Varying Motions

核心思想：物理世界的运动可以自然地分解为瞬态变化和运动趋势。瞬态变化可以看成是各局部区域瞬间的变形、耗散、速度变化或其他变化。如图1所示，一个人在跑步时，身体的不同部位会随着时间的推移发生各种短暂的运动变化，如左腿和右腿交替前进。此外，自然时空过程也呈现出一定的趋势，尤其是物理运动。在图1的运行场景中，身体在每一个时间步上上下摆动，而男子则按照不变的趋势从左向右移动。运动遵循视频序列中物理世界背后的特征，如物体的惯性、雷达回波的气象学或其他物理定律，这些都可以看作是视频的运动趋势。考虑到运动的分解，我们应该捕捉运动的瞬态变化和运动趋势，以便更好地进行时空变化的运动预测。

#### MotionRNN

![image-20220128150445086](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220128150445086.png)

MotionRNN的总体架构如图所示，其中基于ConvLSTM构建的RNN框架中第l层的更新公式如下：

![image-20220128150814597](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220128150814597.png)

其中最后一个方程是运动高速公路，其用之前的隐藏状态$H_t^l$来补偿预测的输出。

值得注意的是，MotionRNN并没有对基本结构做出改变，因此其可以很方便的应用在之前提出的结构上，如ST-LSTM，MIM等。

#### MotionGRU

![image-20220128151713312](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220128151713312.png)

MotionGRU结构如上所示，其中蓝色部分是用于捕捉瞬时变化的$F_t^{'}$，红色部分是趋势动量$D_t^l$对于运动趋势的累加。

其中$F_t^{'}$更新公式如下：

![image-20220128152053277](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220128152053277.png)

其中$u_t、r_t$分别是GRU中的更新门和重置门；$z_t$是当前时刻经过重置之后的特征；Enc对来自上一个预测块的输入进行编码。之后利用更新们$u_t$对上一时刻的瞬时变化特征和当前时刻学到的特征进行结合。

而$D_t^l$的更新规则如下：

![image-20220128153056698](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220128153056698.png)

我们使用$F_{t-1}^l$作为对当前运动趋势的估计，利用近似动量累计的方法可得到如上公式。

通过这种方式，$D_t^l$可以看作是对一段时间内的$F$的加权和，从而可以学到长期的运动趋势特征。

整体更新公式如下：

![image-20220128153347379](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220128153347379.png)

其中$F_t^l$作为对于运动特征的学习器（motion filter），基于运动可分解为瞬时变化和长期运动趋势这一基础，将两者直接相加结合。

其中Warp(·)表示双线性插值的Warp操作，如下图所示：

![image-20220128153850481](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220128153850481.png)

这种操作可以将学习到的运动特征加入到隐藏状态中。

最终的输出$X_t^l$是通过门控$g_t$控制的两个元素的加权和，其中Dec代表Decoder，其解码的特征是已经经过Warp操作的隐藏状态。

由于MotionRNN可以更好的对运动进行建模，可以在状态转换中对运动状态进行编码，可以更好地解决状态转换过程中运动消失的问题，具体表现为对运动序列的建模在长期预测中更加清晰，模糊化程度降低。
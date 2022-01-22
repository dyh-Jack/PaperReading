#### ACNet：Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks

![image-20220115121640031](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220115121640031.png)

基本思路：将每个$d\times d$的卷积核替换为三个分别为$d\times d,1\times d,d\times 1$的卷积核，构成ACNet的基本单元，在卷积计算后将这些结果线性相加即可（卷积操作满足线性相加）。而在推理阶段，将原先的$1\times d,d\times 1$卷积核直接加在$d\times d$的卷积核上，从而形成骨架增强的卷积核。

贡献：

- 可方便快捷的将CNN的原有卷积核替换为新的ACB，并且不会引入新的超参数
- 证明了标准平方卷积核中骨架的重要性
- ACNet可以增强模型对于旋转畸变的鲁棒性
- 没有引入新的自定义结构，未来可方便的对模型进行压缩

在此之前，相关工作已经证明$d\times d$的卷积核可以分解为$d\times 1,1\times d$的卷积核，但对于核的这种变换会产生显著的信息损失

传统CNN架构中，设定卷积核F为$R^{H\times W\times C}$，其中H,W代表图像宽度，C代表通道数，并且这样的卷积核共有D个；那么输入M设定为$R^{U\times V\times C}$,U,V为图像原本的大小，C为通道数；输出O设定为$R^{R\times T\times D}$,R,T代表输出图像大小，并且输出图像有D个特征。

![image-20220115123146077](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220115123146077.png)

在引入BN后，将得到如下式子：

![image-20220115123300072](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220115123300072.png)

其中，$\mu_j,\sigma_j$为样本的均值和标准差，$\gamma_j,\beta_j$为需要学习的标度因子和偏置项。

而对于2D卷积，满足可加性：由于卷积操作的性质（滑动窗口计算），当两个在进行卷积时对应点的窗口相同，即可满足可加性

![image-20220115123511657](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220115123511657.png)

其中I是一个矩阵，K1,K2为两个尺寸**相容**的二维卷积核，右边式子代表将两个二维卷积核对应位置上的元素相加后再进行卷积操作。

而在这里，相容代表我们可以将较小的卷积核叠加在较大的卷积核上

eg. $1\times 3,3\times 1$和$3\times 3$的卷积核是相容的

而在训练之后，在推理阶段，可通过BN融合和分支融合的思想将三个卷积核合为一个标准的卷积核

- BN_fusion 通俗来讲，即将系数带入括号内计算，从而将BN层融合
- branch_fusion 通俗来讲，即将融合后的三个卷积核根据可加性，得到新的卷积核和偏置b

![image-20220115124445906](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220115124445906.png)

![image-20220115124458421](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220115124458421.png)

值得说明的是，在训练阶段三个卷积核不能进行融合，主要原因是权重的随机初始化以及梯度更新的不同。

#### 主要机制

ACNet可以看作骨架增强的标准卷积核，因此这意味着骨架在卷积核中可能占有重要地位。

通过对于ResNet和ACNet进行不同位置的稀疏操作，得到如下图象：

![image-20220115125424560](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220115125424560.png)

由此可明显观察到，去除骨架部分的权重对于神经网络的准确性有着更重要的影响。

之后，通过对于权重的可视化处理，我们可以得到在一个卷积核内部，经过训练后各部分的权重值

![image-20220115125601874](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220115125601874.png)

其中Normal采取ResNet的卷积核作为baseline。

可以看到，ResNet中卷积核的骨架权重相较于边角权重更大，而在ACNet中这种现象得到了进一步的增强。而由于不同卷积核在训练时是独立进行的，理论上每一个位置的卷积核的权重在相加时都可能会被增强或减弱，但模型在训练结束后，都得到了骨架增强这一一致性的结果。因此可推测，对于骨架结构的增强使得ACNet获得了更好的表现性能。

与此同时，我们也设计了边缘增强的ACNet，可以观察到，虽然边缘权重得到了增强，但在实际实验中，当去除骨架结构时带来的性能损失依旧比去除边缘结构更加明显。因此，可以推测，骨架结构在卷积核中具有重要作用。


#### GSM-SGD:Global Sparse Momentum SGD for Pruning Very Deep Neural Networks

- momentum SGD

动量是与SGD一同使用的一种技术，通过累积过去的梯度，而不是仅仅使用当前的梯度，从而使模型能够快速通过局部极小值点，加快收敛速度和效果

![image-20220116131436953](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220116131436953.png)

其中，$\beta$为动量系数，$\eta$为常规的权值衰减系数

定义压缩比C为

![image-20220116132056725](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220116132056725.png)

其中，上半部分代表模型参数的集合，下半部分代表模型中非零参数的个数

因此，需要进行一种权衡，即在保证模型精度没有受到不可接受的削弱的同时，使用更少的参数

由此，即可得到我们的目标函数$L(X,Y,\theta)$

![image-20220116132345445](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220116132345445.png)

其中，$g_i$是指示函数

![image-20220116132529505](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220116132529505.png)

$l_i$是第$i$层所需的非零参数的个数

通常会在训练过程中引入一些额外的可微的正则项来减少一些参数的大小，从而在去除这些参数时对模型的影响更小，因此问题转化为：

![image-20220116132846730](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220116132846730.png)

其中$R(\theta)$是与值相关的正则项，而$\lambda$则是权衡系数

这种方法会带来两个问题：

1. 无法从根本上将集合$\theta$中的某一参数归零，这样在删去这个参数时，将不可避免的带来性能的损失（易得SGD无法将某一参数归零）
2. 超参数$\lambda$无法直接反应最终模型的压缩比，因此如需获得特定压缩比的模型，可能需要在之前进行多次实验从而获得一定的先验知识

- 为此，引入$Q = |\theta|/C$来表示非零元素的数目，并引入新的函数

![image-20220117221659167](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220117221659167.png)

对目标函数泰勒展开可得

![image-20220117221727656](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220117221727656.png)

![image-20220117221737166](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220117221737166.png)

由此，可将SGD的更新公式重写为

![image-20220117221809472](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220117221809472.png)

其中$B^{(k)}$满足

![image-20220117221912974](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220117221912974.png)

对于任意Q都可以进行相应的剪枝，且当C等于1时，GSM退化为普通的动量SGD

由于GSM与模型无关，因此其可用于任意模型的剪枝操作

- 此外，我们还需计算在经过多少次迭代后其中某一个特定的权重可以被剪枝，也就是我们需要知道经过k次迭代后某一权重的数值变为多少。在实际中，我们可以使用如下公式进行估计

![image-20220117222521965](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220117222521965.png)

在固定$w^{(0)}=1,\alpha = 5\times 10^{-3},\eta = 5\times 10^{-4}$时，可得到一些经验结果

![image-20220117222651178](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220117222651178.png)

因此，文章主要思路为：

- 利用Momentum的SGD，使得大部分参数在梯度更新的过程中都趋向于0，并且由于动量的累计，得以加速这一过程，从而显著加快归零的过程。
- 在部分参数归零后，对这些参数进行剪枝，这可以保证在减少参数的同时不会对模型精度带来太大的影响。
- 而主动更新的部分则利用计算得到的梯度进行更新，来保持模型的精度

核心代码如下：

![image-20220117223953439](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220117223953439.png)

其中，mask即为$B^{(k)}$,$all\_g\times all\_v$为参数T
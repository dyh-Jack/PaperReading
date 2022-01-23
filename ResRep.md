#### ResRep: Lossless CNN Pruning via Decoupling Remembering and Forgetting

传统的CNN框架中，学习和遗忘（加入正则项）是集合在卷积核当中的。但生物的记忆和遗忘在大脑中由不同的蛋白质控制，也即进行了记忆和遗忘功能的解耦。受到这一启发，我们实现了ResRep，其中Res代表梯度重置，即一种遗忘规则；Rep代表卷积结构重参数化。

具体来说，就是在每一个conv层接BN层之后插入一个1x1的卷积核（压缩器），这个卷积核用来实现遗忘功能：选择其中的一些通道，并将这些通道中从目标函数获得的梯度置零。在训练过后这些通道的参数将十分接近于0，从而可通过变换将原有的conv和压缩器一起转换为一个通道数更少的conv。其流程伪代码如下：

![image-20220123233319939](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220123233319939.png)

传统CNN的卷积公式如下：

![image-20220123234640287](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220123234640287.png)

设经过剪枝后还存在的通道为$S(i)$，那么经过剪枝后的卷积核为：

![image-20220123235303038](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220123235303038.png)

对于conv_bn层之后的压缩器，即1x1卷积核，我们将这一层的参数展开成$Q\in R^{D\times D}$，其中，两个D分别代表输出通道数和每一个卷积核的参数（此处假设输入的图像通道数为1，经过前端3x3的卷积核后输出通道数为D）。经过Res更新后并剪枝得到的压缩器$Q^{'}\in R^{D^{'}\times D}$，再将其与前面的conv_bn重参数化即可。

首先，conv层和bn层的合并参数易得：

![image-20220124000252265](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220124000252265.png)

之后，我们需要寻找能够将压缩器合并进来的等式，即满足：

![image-20220124000327018](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220124000327018.png)

由于卷积的可加性，可得：

![image-20220124000342304](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220124000342304.png)
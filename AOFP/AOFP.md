#### Approximated Oracle Filter Pruning for Destructive CNN Width Optimization

- Channel Pruning/filter pruning

优点：

1. 具有普适性，可以处理任何类型的CNN
2. 有效性：有效减少网络的浮点运算（FLOPs），而FLOPs是衡量计算量的重要指标
3. 正交性：其没有定制的特殊结构，因而可与其他模型压缩技术并行使用

Channel Pruning的常见范式是根据某种方式评估卷积核的重要性，并将不太重要的卷积核去除

- Oracle Pruning（贪心算法）

对于特定的一层，先去除一个卷积核，然后在评估数据集上测试模型，记录准确率的平均下降数值，之后再恢复当前卷积核，并对下一卷积核进行相同操作，直到完成所有卷积核的测试，并按顺序删除对准确率影响最小的一些卷积核。

这种方法的问题在于，时间复杂度过高，且每次删除一组卷积核之后，由于其他卷积核此时的重要性会发生变化，因此都需要重新开始测试。

此外，这并不是一种启发式方法，因此需要人工设置一部分与其相关的超参数。

引入参数$P^{(i)} = (K^{(i)},b^{(i)})$来表示第i层的参数

![image-20220118214631706](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220118214631706.png)

![image-20220118214644792](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220118214644792.png)

引入参数T来表示卷积核的重要性分数，对于任意一个卷积核F，以L表示目标函数的值

![image-20220118214813638](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220118214813638.png)

由于这样的时间复杂度过高，因此产生了根据删除卷积核所在层的下一层的输出来近似判断卷积核重要性的思路，即

![image-20220118215246716](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220118215246716.png)

![image-20220118215335017](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220118215335017.png)

其中$M_F^{(i)}(x)$是在没有F的情况下第i层的输出

![image-20220118215342983](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220118215342983.png)

整体流程如图所示：

![image-20220118215614338](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220118215614338.png)

其中左边为**基路径**，引入一个掩码因子$u^{(i)}\in R^{c_{i}}$，用于记录已经剪枝的卷积核，则下一层的输出变为

![image-20220118220005048](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220118220005048.png)

$u_k^{(i)} = 0$相当于在这一层中将第k个卷积核去掉

右边为**计分路径**，引入因子$v^{(i)}$。在训练过程中，我们随机使还未被剪枝的卷积核中部分的$v^{(i)}$等于0，相当于在计分路径上将其剪枝。之后计算t，并将其存储起来。如果t很大，说明这个卷积核对于整体结构很重要。

在采集足够多的样本后，我们可以得到

![image-20220118220505241](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220118220505241.png)

其中$\tau^{(i,j)}$是一个集合，记录了所有去掉第i层第j个卷积核得到的T值。

之后我们可以选择g个分数最低的并将其剪枝(即将$u_k^{(i)}$固定为0)，且这样的方法可以在利用i+1层对i层剪枝的同时，利用i+2层对i+1层剪枝。

但此时依旧存在问题：

1. 删除卷积核后网络恢复需要时间，此期间获得的T值不能准确反映剩余卷积核的重要性
2. 超参数g难以求得最优解
3. 何时停止剪枝是启发式的，因此无法知道最终剩余卷积核的数量

因此我们引入二分删除的方法，每次取原先集合中的一半，并不断计算，直到集合中只剩下一个元素或者可被删除的卷积核集合中所有卷积核对于模型的精度影响之和较小，此时删除卷积核；如集合只剩下一个元素且对模型影响依然较大，则停止剪枝。具体算法如下：

![image-20220118220915823](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220118220915823.png)

在此我们引入剪枝阈值$\theta$，我们说集合B足够好，当其满足：

![image-20220118221842063](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220118221842063.png)

与此同时，g的值也得以确定，我们只需将g设定为每次搜索空间的二分之一即可。

![image-20220118223019355](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220118223019355.png)

计算前一层的t值的方法

训练过程：

![image-20220118224632617](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220118224632617.png)

![image-20220118230021392](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220118230021392.png)

核心判断部分：对应与之前的伪代码

![image-20220118225944983](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220118225944983.png)
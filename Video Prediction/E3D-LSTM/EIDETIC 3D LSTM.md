#### EIDETIC 3D LSTM: A MODEL FOR VIDEO PREDICTION AND BEYOND

#### E3D-LSTM

3D卷积被引入RNN被认为可以取得更好的效果，其中我们将(a)(b)作为基线模型，并设计了E3D-LSTM。

![image-20220204221644891](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220204221644891.png)

我们发现在LSTM单元外集成3D-Convs的表现明显不如基线RNN模型。为此，我们提出在LSTM单元内“更深层次”地集成3D-Convs，以便将卷积特征纳入随时间变化的循环状态转换中。

![image-20220204222424735](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220204222424735.png)

E3D-LSTM结构如上图所示，其中右侧的红色箭头代表短期信息流，而蓝色箭头代表长期信息流。其中，X,H,C,M与ST-LSTM中的含义相同。

我们使用循环的3D-Convs作为运动感知器来提取运动的短期外观和局部运动，所得到的内容编码在$R^{T\times H\times W\times C}$的向量中，其中T代表时间深度。

为了捕捉长期框架的交互作用，我们引入了新的记忆状态递归函数如下：

![image-20220204223002032](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220204223002032.png)

其中·代表矩阵乘法，计算时将$R_t,C^k_{t-\tau:t-1}$分别重构为$R^{THW\times C}，R^{\tau THW\times C}$。

![img](http://inews.gtimg.com/newsapp_match/0/10038743572/0)

第一项对局部视频的外观和动作进行编码，其中$I_t$为输入门，$G_t$为标准LSTM一样的输入调制门。第二个$C^k_{t−1}$可以看作是前一个内存状态的捷径连接，它捕获相邻时间戳之间的短期变化。在这个过程中，可访问的内存字段是固定的和有限的。因此，我们引入记忆转换函数的第三项，根据局部运动和外观(以$X_t,H^k_{t−1}$编码)建模长期视频关系。作为对于计算量与效果的平衡，我们通常可将$\tau$设为5

**<u>其中RECALL函数更像是一种注意力机制</u>**

同样，我们也可以将M进行同样的转换：

![image-20220204224446344](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220204224446344.png)

但实验结果表明，对于M的转换对于结果的影响并不明显。

#### 自监督辅助学习

我们考虑两个任务：像素级帧预测和视频分类任务，对于像素级帧预测，有损失函数：

![image-20220204224829623](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220204224829623.png)

![image-20220204225006198](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220204225006198.png)

而对于视频分类任务，我们在训练中使用多任务学习，使这两个任务共享同一个目标函数：

![image-20220204225237855](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220204225237855.png)

其中$L_{ce}$是交叉熵函数，$\lambda$为权重因子。

尽管改善这两个任务需要适当的长期短期情境表征，但不能保证通过像素级监督学习到的特征将完全符合任何高级目标。因此，我们引入了一种计划性学习策略，目标函数逐渐从一个任务向另一个任务倾斜。具体来说，我们在迭代次数i上对λ应用线性衰减：

![image-20220204225426922](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220204225426922.png)

其中$\lambda(0),\eta$分别为$\lambda(i)$的最大值和最小值，$\epsilon$为控制衰减速度的参数。我们将这种方法称为自监督辅助学习。
#### DQN&Double DQN

##### Q-Learning

Q-Learning中最为重要的公式如下：

![image-20220322160735592](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220322160735592.png)

其本质上是一个滑动平均的过程：也就是说，对于任意一个时间步下，选择动作a转移到$s^{'}$之后的预期值函数用$r+\gamma maxQ(s^{'},a^{'})$来表示，当遍历当前状态s所有可能的动作a和下一状态$s{'}$时，此时得到的Q值是一个较为准确的估计，而公式做的就是对这些所有的情况进行滑动平均，从而可以迭代得到较为准确的值函数估计。

而对于Q-Learning，有两种采样方式：

![image-20220322161311483](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220322161311483.png)

![image-20220322161323657](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220322161323657.png)

Q-Learning的思想与TD,或者说是SarSa相近，区别是利用max替换了SarSa中采用原有a的方式。

##### DQN

回到DQN，在值函数近似中，我们通常都利用我们估计的target来代替真实的值函数，也就是损失函数中的真实值。而在这里，target就是$r+\gamma maxQ(s^{'},a^{'})$。因此我们很容易得到损失函数：

![image-20220322161652521](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220322161652521.png)

而与Q-Learning不同的是，Q-Learning中用来计算target和预测值的Q是同一个Q，也就是说使用了相同的神经网络。这样带来的一个问题就是，每次更新神经网络的时候，target也都会更新，这样会容易导致参数不收敛。因此DQN在原来的Q网络的基础上又引入了一个**target Q网络**，即用来计算target的网络。它和Q网络结构一样，初始的权重也一样，只是Q网络每次迭代都会更新，而target Q网络是每隔一段时间才会更新。（**Fixed Q Target**）

##### Double DQN

DQN有一个显著的问题，就是DQN估计的Q值往往会偏大。这是由于我们Q值是以下一个s'的Q值的最大值来估算的，但下一个state的Q值也是一个估算值，也依赖它的下一个state的Q值...，这就导致了Q值往往会有偏大的的情况出现。

因此，Double DQN的核心思路就是引入两个Q网络，**Q1网络推荐能够获得最大Q值的动作；Q2网络计算这个动作在Q2网络中的Q值**

而由于DQN中本身就有两个Q网络，因此我们在这里将原始DQN中用于计算Q的网络看作Q1，用于计算target的网络看作Q2，则Double DQN的核心不同：

![image-20220322163614615](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220322163614615.png)
#### PredRNN++: Towards A Resolution of the Deep-in-Time Dilemma in Spatiotemporal Predictive Learning

PredRNN在提出后仍然面临许多问题，其中最为突出的便是其作为具有时空记忆流的LSTM和传统LSTM的直接结合，在反向传播中依然面临着回传路径过长，从而导致梯度消失的问题（具有时空记忆流的LSTM回传路径过长）

![image-20220123171013916](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220123171013916.png)

#### 因果LSTM

![image-20220123172102459](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220123172102459.png)

其中$*$代表卷积，圆中心一点代表逐个元素相乘，方括号代表张量的串联，$W_1-W_5$为卷积核，其中，$W_3、W_5$为用来改变输出通道数的1x1卷积核，最终输出的隐藏状态$H_t^k$由$C_t^k,M_t^k$共同决定

![image-20220123172802240](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220123172802240.png)

因果LSTM结构如上，其中同心圆代表张量的串联

![image-20220123173412068](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220123173412068.png)

这是一个很明显的三层级联的结构。其特点如下：

1. 每个门不是由X和H决定，而是由X和H以及C决定，通过输入门之前的状态也是又三者决定的

2. 每个cell都是先进行输入门和忘记门的操作，之后输出这个cell的cell state，把输出门的结构放在了整个Causal LSTM的最后的橘色第三部分。

3. 依旧存在两个memory结构，即C和M，只不过这里明显区分，C为temporal state，M为spatial state，因为输入C为上一个时刻的C，M是上一层的M，所以这里C与时间维度有关，M与空间维度有关。

4. M作为第二部分的state输入，并且通过忘记门之前做了一个非线性操作tanh，

5. 这里的输出门不是用的sigmoid激活，而是采用tanh激活，并且输出门的输入为前两个级联cell的输出以及最开始的输入

6. 最后输出H取决于两个cell的共同输出即此时cell的temporal state和spatial state，而不是第二部分的最终输出state。

**对比ST-LSTM来说，Causal LSTM对于M和H定义更加清晰，并且不是简单的concat，而是采用了一个递归深度更深的一个级联结构最终输出H**

#### 梯度公路单元

为了解决梯度消失的问题，引入梯度公路单元，更新规则如下：

![image-20220123174303290](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220123174303290.png)

其中W为卷积核，$S_t$被称为开关门（Switch Gate），更新规则可简单描述为$Z_t = GHU(X_t,Z_{t-1})$

![image-20220123174728252](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220123174728252.png)

#### PredRNN++

利用上述两个结构，可以构建更深层次，建模能力更好的网络架构。

**具体来说，我们将L层因果lstm叠加，并在第1个和第2个因果lstm之间注入一个GHU。**整体更新规则如下：

![image-20220123174818731](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220123174818731.png)

![image-20220123174832505](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220123174832505.png)

在整体架构中，梯度公路单元和因果LSTM单元分别负责建模长期和短期的视频依赖关系，使学习能够获得长期和短期的特征。

代码思路十分清晰，且架构与PredRNN的代码架构相似
#### PredRNN：Recurrent Neural Networks for Predictive Learning using Spatiotemporal LSTMs

![image-20220114133148196](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220114133148196.png)

时空序列预测的目标公式，即给定J个序列，预测未来K个序列的最可能的结果

![image-20220114133519176](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220114133519176.png)

卷积LSTM的关键方程，其中$g_t,i_t,f_t,o_t$均为LSTM的门控单元，$C_t,H_t$分别为卷积LSTM单元的输出和隐藏状态。

其中所有变量都是一个$R^{P \times M\times N}$的向量，其中M,N表示图像的行和列，P代表输入的图像数目或特征数量。

在这之中，sigma为sigmoid激活函数，*代表卷积运算，圆圈加点代表哈达玛积（即$c_{ij} = a_{ij}\times b_{ij}$）。

![image-20220114135214858](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220114135214858.png)

左图为具有时空记忆流的LSTM结构，右图为传统的卷积LSTM结构

对于有监督学习，右图结构是合理的，因为根据对叠卷积层的研究，隐藏表示可以从底层向上变得越来越抽象和类特定。然而，我们假设在预测学习中，需要保持原始输入序列中的详细信息。如果我们想看到未来，我们需要从不同层次的卷积层提取的表示中学习。

![image-20220114135454737](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220114135454737.png)

对于左图使用的LSTM架构,即具有时空记忆流的LSTM，其关键方程如上。

与传统的LSTM（如右图所示）不同，传统LSTM结构在同一层（横向上），每一个单元的输入依赖于上一个时刻同一位置上的隐藏状态，即$H_{t-1}^{l},C_{t-1}^{l}$，而左图的LSTM结构，除顶层外，每一层的输入依赖于$H_{t}^{l-1},C_{t}^{l-1}$

<u>然而，时空记忆流的LSTM在水平方向上降低了时间流，从而牺牲了一定的时间相干性。</u>

![image-20220114141351153](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220114141351153.png)

ST-LSTM的关键方程如上。

ST-LSTM融合了经典的LSTM架构，同时也引入了具有空间记忆流的LSTM结构，每个单元中维护了两个内存单元，而每个单元最终的输出和隐藏状态都取决于这两部分的结合。在使用时，可以将来自不同方向的内存串接起来，再利用1x1卷积进行降维，**从而保证输入输出维数相同**。

![image-20220114140830100](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220114140830100.png)

$ST-LSTM(左)和PredRNN(右)$。

ST-LSTM单元中的橙色圆圈表示与传统$ConvLSTM$的差异。PredRNN中的橙色箭头表示时空记忆流，即左侧时空记忆$M_{t}^{l}$的过渡路径。

#### 实验设置

实验使用L1+L2损失，利用adam优化器训练，初始学习速率设置为$10^{-3}$，共迭代80000次，batch size设置为8.

实验中，每个序列由20个连续帧组成，10为输入帧，10为预测帧。

实验利用手写数字进行图像预测。

对于每个数字，我们指定一个速度，其方向是在单位圆上均匀分布随机选择的，幅值是随机选择的。当到达同一位置时，数字会从图像的边缘反弹并相互遮挡。这些特性使得在不了解运动的内部动态的情况下，模型很难给出准确的预测。

#### 代码实现

- Layer Normalization

与BN不同，BN是取不同样本的同一个通道的特征做归一化；LN则是取同一个样本的不同通道做归一化。

由于在动态的RNN网络中，数据较少，而BN的归一化依赖于样本的数目，在样本较少时BN的归一化效果较差，因此对于RNN动态网络引入Layer Normalization

对于MLP中的LN，设H为一层中隐藏节点的个数，则LN的参数$\sigma,\mu$可由如下公式给出。

![image-20220114144046462](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220114144046462.png)

<u>注意上面统计量的计算是和样本数量没有关系的，它的数量只取决于隐层节点的数量，所以只要隐层节点的数量足够多，我们就能保证LN的归一化统计量足够具有代表性。</u>

可得到归一化的数值如下

![image-20220114144234802](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220114144234802.png)

此外，我们也需要一组参数来保证归一化操作不会破坏之前的信息，在LN中这组参数叫做增益g 和偏置b，相当于BN中的$\beta,\gamma$。设激活函数为f，可得输出为

![image-20220114144428826](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220114144428826.png)

合并后可得

![image-20220114144440129](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220114144440129.png)

对于RNN，设其t时刻总的输入为$a^t$，$x^t$为t时刻的输入，则

![image-20220114144634011](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220114144634011.png)

归一化LN为

![image-20220114144727821](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220114144727821.png)

- SpatioTemporalLSTMCell实现

<img src="C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220114150443217.png" alt="image-20220114150443217" style="zoom:200%;" />

其中conv_x,conv_h,conv_m,conv_o相当于对于所有与这一变量相关的矩阵直接进行卷积计算，之后的num_hidden所乘系数不同，即是在对这些数据进行分割时使用。

![image-20220114150646598](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220114150646598.png)

进行数据分割，从而得到用于不同部分的运算项

其中torch.split使用如下

![image-20220114151335361](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220114151335361.png)

<img src="C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220114150721992.png" alt="image-20220114150721992" style="zoom: 80%;" />

将不同运算项按照公式进行组合，最终返回当前单元的H,M,C

- PredRNN架构

变量：num_layers，即一共几层，在PredRNN中，一个纵向上的LSTM单元数为其层数

​			num_hidden，即隐藏层的通道数

![image-20220114154147369](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220114154147369.png)

上述部分相当于构建了一个完整的纵向单元，从输入原始图像到顶端输出

- Scheduled Sampling

原本的采样方法，训练时该模型将目标序列中的真实元素作为解码器每一步的输入，然后最大化下一个元素的概率。生成时上一步解码得到的元素被用作当前的输入，然后生成下一个元素。<u>可见这种情况下训练阶段和生成阶段的解码器输入数据的概率分布并不一致。</u>

Scheduled Sampling是一种解决训练和生成时输入数据分布不一致的方法。在训练早期该方法主要使用目标序列中的真实元素作为解码器输入，可以将模型从随机初始化的状态快速引导至一个合理的状态。随着训练的进行，该方法会逐渐更多地使用生成的元素作为解码器输入，以解决数据分布不一致的问题。

标准的序列到序列模型中，如果序列前面生成了错误的元素，后面的输入状态将会收到影响，而该误差会随着生成过程不断向后累积。Scheduled Sampling以一定概率将生成的元素作为解码器输入，这样即使前面生成错误，其训练目标仍然是最大化真实目标序列的概率，模型会朝着正确的方向进行训练。因此这种方式增加了模型的容错能力

因此，Scheduled Sampling主要应用在序列到序列模型的训练阶段，而生成阶段则不需要使用。

训练阶段解码器在最大化第t个元素概率时，标准序列到序列模型使用上一时刻的真实元素$y_{t−1}$作为输入。设上一时刻生成的元素为$g_{t−1}$，Scheduled Sampling算法会以一定概率使用$g_{t−1}$作为解码器输入。

设当前已经训练到了第i个mini-batch，Scheduled Sampling定义了一个概率$ϵ_i$控制解码器的输入。$ϵ_i$是一个随着i增大而衰减的变量

常见的三种衰减方式曲线如下

<img src="C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220114154630866.png" alt="image-20220114154630866" style="zoom:80%;" />

![image-20220114154820755](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220114154820755.png)

不同的采样方法决定每一个纵向上起始单元的输入

run.py中定义了不同的采样方法，其中引入了**Reverse Scheduled Sampling**，即先让模型使用前一单元生成的内容进行训练，再逐渐过渡到使用ground truth进行训练。好处在于这将促使模型更好的学习一些长期运动特征

![image-20220114154903749](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220114154903749.png)

在每一个纵向方向上进行循环，最终在每一个纵向方向上得到下一个时刻的预测结果

- argparse

argparse主要用于命令行选项与参数解析，通常分为3步

1.创建一个解析器

```python
parser = argparse.ArgumentParser(description='PyTorch video prediction model - PredRNN')
```

2.添加参数

```python
parser.add_argument('--is_training', type=int, default=1)
```

![image-20220114160205330](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220114160205330.png)

3.解析参数

```python
args = parser.parse_args()
```
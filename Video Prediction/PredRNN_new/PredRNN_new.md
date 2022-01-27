#### PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning

RNN的目标函数如下：

![image-20220125223402661](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220125223402661.png)

传统的卷积LSTM更新公式如下：

![image-20220125223532347](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220125223532347.png)

其可能面临如下问题：

1. 记忆状态Ct仅仅沿着时间箭头在每个ConvLSTM层更新，较少依赖于其他层的分层视觉特征。因此，当前时间步长的第一层可能会在很大程度上忽略上一时间步的顶层所记忆的内容。
2. 在每个ConvLSTM层中，输出隐藏状态Ht依赖于记忆状态Ct和输出门ot，这意味着记忆单元被迫同时应对长期和短期动态。因此，记忆状态转移函数对复杂时空变化的建模能力可能会极大地限制预测模型的整体性能。
3. ConvLSTM网络遵循序列到序列rnn的架构。在训练阶段，它总是以真实的上下文框架作为编码时间步长的输入，这可能会损害长期动态的学习过程。

而PredRNN的提出很好的解决了以上问题。

![image-20220126170343900](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220126170343900.png)

其中，PredRNN构建了具有时空记忆流的LSTM结构，其更新公式如下：

![image-20220126170328298](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220126170328298.png)

其中，保留经典LSTM的结构，得到PredRNN，从而能够使得模型对于长期和短期结构的建模能力都得到提升

![image-20220126173608078](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220126173608078.png)

![image-20220126173621699](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220126173621699.png)

PredRNN有如下优点：

1. M引入了更深的过渡路径，在ST-LSTM单元之间曲折前进。它提高了从一个时间步到下一个时间步建模复杂短期动态的能力，并允许H以不同的速率自适应传输。
2. C在较慢的时间尺度上运行。它提供了较短的隐藏状态之间的梯度路径，从而促进了长期依赖的学习过程。

而在实际应用中，我们发现很多情况下C与M都会纠缠在一起，很难通过它们各自的网络架构自发地解耦。这在一定程度上导致了网络参数的低效利用。因此，我们在第一版的基础上，提出了进一步的解耦结构：我们首先在每个时间步长的Clt和Mlt增量上添加卷积层，并利用新的解耦损失在潜空间中显式地扩展它们之间的距离。通过这种方法，不同的记忆状态被训练成关注于时空变化的不同方面。内存整体解耦方法可以表述为:

![image-20220126194939077](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220126194939077.png)

值得注意的是，新的参数仅在训练时使用，在推理时从整个模型中删除。即与之前的ST-LSTM版本相比，模型尺寸没有增加。通过用余弦相似度定义解耦损失，我们的方法鼓励两个存储状态在任何时间步长的增量是正交的。它释放了C和M在长期和短期动态建模方面各自的优势。（其中$L_{decouple}$公式中上方代表点积，下方代表范数）

因此，总的损失函数为：

![image-20220126195323518](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220126195323518.png)

此外，我们提出了新的采样方法，即Reverse scheduled sampling：

![image-20220126195554867](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220126195554867.png)

在经典训练方法中，在时间步小于给出的ground truth时间步时都使用真实输入，之后使用预测数据作为输入，即：

![image-20220126195800973](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220126195800973.png)

而对于RSS，我们有如下训练方案：

![image-20220126195914198](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220126195914198.png)

其中RSS，SS分别代表Reverse Scheduled Sampling 和 Scheduled Sampling

![image-20220126200103025](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220126200103025.png)

即RSS初始更倾向于使用预测的图片作为下一个时间步的输入，在后期逐渐使用ground truth作为输入。这将是其逐渐从生成多步未来帧(由于缺乏一些历史观测而具有挑战性)转变为一步预测，就像编码器在测试时所做的那样。这样的训练方案鼓励模型从输入序列中提取长期的、非马尔可夫动力学。

其中对于$\epsilon_k$有如下参数衰减方案：

![image-20220126200148218](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220126200148218.png)

在实际训练中我们往往将RSS与SS结合起来使用（公式中同时涉及到了RSS与SS），其中上图右侧的方案效果更好。
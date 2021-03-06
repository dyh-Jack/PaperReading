#### Memory In Memory: A Predictive Neural Network for Learning Higher-Order Non-Stationarity from Spatiotemporal Dynamics

#### 基本知识

数据中存在平稳信息与非平稳信息，但主要的难点在于非平稳的预测，因为非平稳基本上是没有规律的，而平稳的信息是存在一定的周期或者规律不变的变换的。

低阶的非平稳性：区域像素点的空间和时间的关系，即小块区域的时空变换。

高阶的变化：例如在气象中的雷达回波图中对应的数据的堆积、变形或耗散，即整体的变换趋势，更为复杂。

任何一个非平稳过程都可以分解为：确定项+时间变量多项式+零均值随机项

而通过差分的操作，我们可以把时间变量多项式转换成一个常量，使确定性的组成部分可预测。这里引入了差分。

大多数的经典时间序列分析也都是假设通过差分的转换方式把非平稳趋势近似的转换为平稳项，比如ARIMA。

对于LSTM有三个门，忘记门，输入门，和输出门，而忘记门是和过去的状态最密切的gate。

在模型的工作过程中，忘记门有百分之80都处于饱和状态，即意味着state的传播是一个时不变的传播，也就是忘记门在工作的时候总是记住平稳的变换信息。 这里近似可以看作对于整个的预测就是个类似于线性的推理，因为经过忘记门的state是不变的，基本上可以说预测只取决于输入，而取决于先前的信息也是固定的。

eg：假设我们有一个视频序列，显示一个人以恒定的速度行走。速度可以看作是一个平稳变量，而腿的摆动应该看作一个非平稳过程，这显然更难预测。不幸的是，之前的LSTM类模型中的遗忘门是一个简单的门控结构，它难以捕捉时空中的非平稳变化。在初步实验中，我们发现在最近的PredRNN模型中，大多数遗忘门是饱和的，这意味着单元总是记住平稳的变化。（**因此，在预测图像中人的位置是清晰的，但腿部的动作可能会模糊**）

#### MIM Block

1. 通过对相邻隐藏状态的差分来实现对时空非平整度的统一建模
2. 通过叠加多个MIM块，我们的模型有机会逐步平稳时空过程，使其更具可预测性
3. 需要注意的是，对于时间序列的预测，过度差分是不好的，它不可避免地会导致信息的丢失。这是我们在记忆转换中应用差分而不是在所有递归信号中应用差分的另一个原因，例如输入门（input gate）和输入调制门（gate gate）
4. MIM有一个来自LSTM的记忆单元，另外还有两个循环模块，它们的记忆嵌入在第一个记忆的过渡路径中。我们利用这些模块分别对时空动态的高阶非平稳分量和近似平稳分量进行建模

![image-20220127213615267](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220127213615267.png)

MIM中将原本ST-LSTM中的遗忘门替换为两个模块，第一个模块（MIM-N）额外以$H_{t-1}^{l-1}$作为输入，用于捕获两个相邻连续的隐藏项之间的非平稳变化（$H_t^{l-1}-H_{t-1}^{l-1}$），它会根据差分平稳假设生成微分特征$D_t^l$；另一个模块（MIM-S）以$D_t^l$和$C_{t-1}^l$为输入，捕获序列中近似平稳的变化。通过将MIM-N和MIM-S级联起来，得到输入$\tau_t^l$，并以此替换ST-LSTM中的遗忘门。

MIM更新公式如下：

![image-20220127215642422](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220127215642422.png)

其中N和S分别代表非平稳模块（MIM-N）和平稳模块（MIM-S）中的水平过渡记忆单元。（用于储存上一个对应单元的部分信息）

![image-20220127215904129](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220127215904129.png)

其中MIM-N和MIM-S的具体计算过程如下：

![image-20220127215946071](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220127215946071.png)

MIM-N中所有更新都与两个相邻连续的隐藏项之间的非平稳变化（$H_t^{l-1}-H_{t-1}^{l-1}$）有关，这突出了时空序列的非平稳变化。

![image-20220127220103902](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220127220103902.png)

MIM的直观解释是：**<u>这两个结构一起做的事就是第一个结构提取出非平稳信息,之后传递给第二个结构，第二个结构利用门控来选择是否忘记和记住多少的非平稳信息或者memory cell（C）即近似的平稳信息。提供了一个门控机制，可以自适应的决定是否信任原始记忆$C_{t-1}^l$或差分特征$D_t^l$；如果差分特征消失，表明非平稳动态不突出，那么MIM-S将着重使用原始记忆；反之，如果差分特征突出，那么MIM-S将更多地关注非平稳特征，将部分原始记忆覆盖。</u>**

#### MIM Net

MIM Net体系结构的关键思想是提供必要的隐藏状态来生成差异特征，并最好地促进非平稳建模。

MIM Net整体结构如下：

![image-20220127221024469](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220127221024469.png)

需要注意的是，**在第一层以及在第一个时间步的时候**，由于无法构成如图所示的差分结构，因此还是需要使用传统的ST-LSTM

整体公式如下：

![image-20220127221245851](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220127221245851.png)

#### 代码实现

代码实现过程中有部分不同：第一个时间步的时候依旧采用MIM，不过做了部分改动：

![image-20220127225803120](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220127225803120.png)


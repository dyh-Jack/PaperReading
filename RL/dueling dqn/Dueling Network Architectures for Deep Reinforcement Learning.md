#### Dueling Network Architectures for Deep Reinforcement Learning

对于传统的PG，我们有以下定义：

![image-20220322152338424](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220322152338424.png)

![image-20220322152403165](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220322152403165.png)

![image-20220322152410984](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220322152410984.png)

![image-20220322152418100](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220322152418100.png)

**从直观上看，价值函数V衡量的是处于某一特定状态的好坏程度。然而，Q函数衡量的是在这种状态下选择某一特定行动的价值。优势函数从Q函数中减去状态值，以获得每个动作重要性的相对度量。**

**<u>所以，其实Dueling DQN的核心思想是在Q值估计中对状态和动作进行解耦</u>**

对于普通Bellman方程，我们可以得到：

![image-20220322153723792](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220322153723792.png)

而对于最优值函数，我们依旧可以得到：

![image-20220322153834454](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220322153834454.png)

所以，Dueling DQN希望设计两个网络来分别学习V和A。也就是说，换句话说，Value Stream表示了这个 ![[公式]](https://www.zhihu.com/equation?tex=s) 的好坏，而Advantage Stream则表示了每个 ![[公式]](https://www.zhihu.com/equation?tex=a) 相对的好坏。

因此，我们在构建网络时可以将目标Q表示为如下形式：

![image-20220322154149178](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220322154149178.png)

可以得到梯度：![image-20220322154207440](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220322154207440.png)

其中$y_i^{DQN}$代表：

![image-20220322154250085](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220322154250085.png)

然而这种训练方式在实际应用中很难保证**收敛的一致性**：因为如果对A加上某个常数C，在V中再减去这个常数C，得到的结果依然相同。

作者为了解决它，作者强制优势函数估计量在选定的动作处具有**零优势**。 也就是说让网络的最后一个模块实现前向映射，表示为：

![image-20220322154555855](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220322154555855.png)

也就是说：对于任意a来说：

![image-20220322154626583](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220322154626583.png)

我们利用上面的公式都可以得到：

![image-20220322154658456](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220322154658456.png)

因此，$V(s;\theta,\beta)$提供了价值函数的估计，而另一个产生了优势函数的估计。相当于此时的A就是对于当前动作与最优动作的差值进行估计，也就是强迫A只估计优势函数，从而保证了收敛的一致性。

而在实际应用中，利用平均代替了最大值操作，表示为：

![image-20220322154832405](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220322154832405.png)

采用这种方法，虽然使得值函数V和优势函数A不再完美的表示值函数和优势函数，但是这种操作提高了稳定性。而且，并没有改变值函数V和优势函数A的本质表示。

在这里，由于之前的推理，max A和E(A)均为0，因此这种操作并不会改变Q的数值，但是会强迫网络具有收敛一致性。
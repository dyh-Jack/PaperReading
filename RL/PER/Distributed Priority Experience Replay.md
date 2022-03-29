#### Distributed Priority Experience Replay

![image-20220329224548286](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220329224548286.png)

![image-20220329224722540](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220329224722540.png)

![image-20220329224731657](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220329224731657.png)

本文实现了一个分布式的$PER$，可以将Actor的行为与Learner的学习区分开。二者的行为伪代码如上所示，**其中Learner的参数实时更新，并且定期将更新的网络参数传递给Actor**。我们据此设计出了Ape-X网络。

与共享梯度相比，共享经验有一定的优势。由于分布式SGD要求低延时通信，而经验相比于梯度，对于低延迟的要求没有那么高，只需要对于off-policy具有鲁棒性。因此，我们可以通过集中式PER对所有经验进行批处理，从而以延迟为代价提高效率和吞吐率。

我们的网络架构综合采用了rainbow、double DQN和Dueling DQN等结构。我们的损失函数如下：

![image-20220329231402776](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220329231402776.png)

![image-20220329231532449](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220329231532449.png)

理论上，Q-Learning是一种off-policy的方法。因此我们可以自由选择用于生成数据的策略。不过，在实践中，我们对策略的选择会影响对于函数逼近的探索质量。此外，由于我们使用的是没有对于off-policy修正的multi-step return，理论上可能会对价值估计产生不利影响。尽管如此，Ape-X DQN中，我们的每个Actor执行不同的策略，并依赖于优先级机制来选取最有效的经验。在实验中，每个Actor使用不同$\epsilon$值的$\epsilon-greedy$策略。其中，低$\epsilon$鼓励在环境中更多进行探索，而高$\epsilon$能够防止over-specialization

#### Analysis

我们通过对比实验发现，仅仅通过增加Actor的数量，而不改变网络参数的更新速率、网络结构或更新规则，性能就能得到明显改善。根据我们的分析，我们的网络体系结构能够解决**一种常见的深度强化学习问题：局部最优。**例如，由于探索不足。使用大量参与者并进行不同程度的探索有助于发现有前途的新行动路线，优先重放可以确保当这种情况发生时，学习算法会将精力集中在这一重要信息上。

我们还发现，提高PER的内存能够在一定程度上提升训练结果。我们认为这是因为更大的PER内存能够使网络保存更多具有高优先级的经验，并通过PER不断学习他们。

此外，对于其他因素的分析并不足以解释我们所观测到的性能。因此，我们得出结论：更好的性能主要来源于收集更多Experience的积极影响，即更好地探索和更好地避免过拟合。

这篇研究表明分布式系统在深度强化学习当中有着巨大的应用前景
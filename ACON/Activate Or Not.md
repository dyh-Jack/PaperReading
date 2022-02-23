#### Activate Or Not

![image-20220222140527253](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220222140527253.png)

我们设计了能够学习是否激活的激活函数单元ACON，能够自适应的决定是否需要进行激活。在所有模型上都得到了稳定的提升。

#### Smooth Maximum

对于最大值函数$max(x_1,x_2,x_3,...)$，其有光滑可微近似：

![image-20220222141607902](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220222141607902.png)

其中$\beta$为切换因子，当$\beta$趋于无穷时，函数值为最大值；当$\beta$趋于0时，函数值为算术平均值。

在神经网络中，许多常见的激活函数都是$max(\eta_a(x),\eta_b(x))$的形式，其中$\eta$表示线性函数，而我们的目标是用上述函数来近似max函数。

考虑n=2的情形，用$\sigma$表示sigmoid函数：

![image-20220222142039138](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220222142039138.png)

当$\eta_a(x) = x,\eta_b(x) = 0$时，此时得到的$S_\beta = x\sigma(\beta x)$，我们称之为ACON A，而这正是Swish的公式。

Swish是NAS搜索得到的一个激活函数，但对于其为何会提高性能，缺乏解释。我们认为Swish是ReLu的一个光滑近似。

接下来我们考虑PReLu的变换：

PReLU的一般形式为：$f(x) = max(x,0)+p\times min(x,0)$，其中p是一个可学习的参数，一般初始化为0.25。在这里我们将PReLU重写为如下形式：$f(x) = max(x,px)$，得到的$S_\beta = (1-p)x \times \sigma[\beta(1-p)x]+px $，即ACON B.

之后，我们给出了一般形式，即ACON C:$S_\beta(p_1x,p_2x) = (p_1-p_2)x\times \sigma[\beta(p_1-p_2)x]+p_2x$

![image-20220222143751454](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220222143751454.png)

#### Upper/Lower bounds

可以证明Swish有固定的上下界，而ACON C具有可学习的上下界

可计算得到其一阶导数与极限如下：

![image-20220222215710722](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220222215710722.png)

二阶导数如下：

![image-20220222215730599](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220222215730599.png)

令其二阶导数等于0，我们可以得到$(y-2)e^y = y+2,y =(p_1-p_2)\beta x$

因此我们可以得到（5）式的最大最小值：

![image-20220222215929648](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220222215929648.png)

#### Meta ACON

开关因子$\beta$以输入x为条件，$\beta = G(x)$

对于不同情况分别由不同的$G(x)$：

![image-20220222220413835](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220222220413835.png)

![image-20220222220423948](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220222220423948.png)

![image-20220222220437475](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220222220437475.png)
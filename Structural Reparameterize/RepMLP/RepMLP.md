#### RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality

- MLP（多层感知机）

![image-20220119142429493](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220119142429493.png)

最简单的MLP只有三层，分别为输入层、隐藏层、输出层

可以看到，**感知机的层与层之间是全连接的**

![image-20220119142553804](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220119142553804.png)

```python
import os
import sysimport time
 
import numpy
 
import theano
import theano.tensor as T
//导入包
```

```python
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
注释：
这是定义隐藏层的类，首先明确：隐藏层的输入即input，输出即隐藏层的神经元个数。输入层与隐藏层是全连接的。
假设输入是n_in维的向量（也可以说时n_in个神经元），隐藏层有n_out个神经元，则因为是全连接，
一共有n_in*n_out个权重，故W大小时(n_in,n_out),n_in行n_out列，每一列对应隐藏层的每一个神经元的连接权重。
b是偏置，隐藏层有n_out个神经元，故b时n_out维向量。
rng即随机数生成器，numpy.random.RandomState，用于初始化W。
input训练模型所用到的所有输入，并不是MLP的输入层，MLP的输入层的神经元个数时n_in，而这里的参数input大小是（n_example,n_in）,每一行一个样本，即每一行作为MLP的输入层。
activation:激活函数,这里定义为函数tanh
        """
        
        self.input = input   #类HiddenLayer的input即所传递进来的input
 
"""
注释：
代码要兼容GPU，则W、b必须使用 dtype=theano.config.floatX,并且定义为theano.shared
另外，W的初始化有个规则：如果使用tanh函数，则在-sqrt(6./(n_in+n_hidden))到sqrt(6./(n_in+n_hidden))之间均匀
抽取数值来初始化W，若时sigmoid函数，则以上再乘4倍。
"""
#如果W未初始化，则根据上述方法初始化。
#加入这个判断的原因是：有时候我们可以用训练好的参数来初始化W
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)
 
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
 
#用上面定义的W、b来初始化类HiddenLayer的W、b
        self.W = W
        self.b = b
 
#隐含层的输出
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
 
#隐含层的参数
        self.params = [self.W, self.b]
```

```python
"""
定义分类层，Softmax回归
在deeplearning tutorial中，直接将LogisticRegression视为Softmax，
而我们所认识的二类别的逻辑回归就是当n_out=2时的LogisticRegression
"""
#参数说明：
#input，大小就是(n_example,n_in)，其中n_example是一个batch的大小，
#因为我们训练时用的是Minibatch SGD，因此input这样定义
#n_in,即上一层(隐含层)的输出
#n_out,输出的类别数 
class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
 
#W大小是n_in行n_out列，b为n_out维向量。即：每个输出对应W的一列以及b的一个元素。  
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
 
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
 
#input是(n_example,n_in)，W是（n_in,n_out）,点乘得到(n_example,n_out)，加上偏置b，
#再作为T.nnet.softmax的输入，得到p_y_given_x
#故p_y_given_x每一行代表每一个样本被估计为各类别的概率    
#PS：b是n_out维向量，与(n_example,n_out)矩阵相加，内部其实是先复制n_example个b，
#然后(n_example,n_out)矩阵的每一行都加b
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
 
#argmax返回最大值下标，因为本例数据集是MNIST，下标刚好就是类别。axis=1表示按行操作。
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
 
#params，LogisticRegression的参数     
        self.params = [self.W, self.b]
```

```python
#3层的MLP
class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )
 
#将隐含层hiddenLayer的输出作为分类层logRegressionLayer的输入，这样就把它们连接了
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
 
 
#以上已经定义好MLP的基本结构，下面是MLP模型的其他参数或者函数
 
#规则化项：常见的L1、L2_sqr
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )
 
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )
 
 
#损失函数Nll（也叫代价函数）
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
 
#误差      
        self.errors = self.logRegressionLayer.errors
 
#MLP的参数
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        # end-snippet-3
```

全连接的问题：**需要展开成一列向量进行运算，从而忽略了空间位置信息**

**但从另一个角度，全连接层也抹去了位置的影响，使得图像识别有更高的鲁棒性**

- 论文思路

定义$M \in R^{n\times c\times h\times w}$，其中n为批量大小，c为通道数，h，w为图像长宽；定义F和W分别为卷积层和全连接层的核

![image-20220119144106442](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220119144106442.png)

其中$M^{(out)}\in R^{n\times o\times h^{'}\times w^{'}}$为输出的特征向量；$F\in R^{o\times c\times k\times k}$为卷积核参数，p为padding。为了方便，假设$w = w^{'},h = h^{'}$

对于全连接层，设p和q分别为输入维度和输出维度

![image-20220119144851221](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220119144851221.png)

$V^{(out)}\in R^{n\times q},V^{(in)}\in R^{n\times p},MMUL$为矩阵乘法

在经过全连接层时，原本上一层的out先被一个变换函数变成$V^{(in)}\in R^{n\times chw}$，而最终的输出其实是$V^{(out)}\in R^{n\times ohw}$

现在假设存在一个与FC平行的conv层，我们讨论如何将其合并进入FC中，即

![image-20220119145551456](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220119145551456.png)

由于对于任何结构与$W^{(1)}$相同的矩阵，其可加性有保证。因此，如果我们能构造$W^{(F,p)}$与其结构相同即可合并：

![image-20220119145742171](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220119145742171.png)

显然，由于**conv层可以视作一个参数稀疏的FC层**，故这样的矩阵肯定存在

![image-20220119145927875](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220119145927875.png)

利用矩阵乘法，可得

![image-20220119145952292](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220119145952292.png)

在其中添加一个恒等变换，并利用结合律

![image-20220119150015104](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220119150015104.png)

将前文提到的变换函数显式定义为RS

![image-20220119150104309](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220119150104309.png)

括号内部可看作一个卷积操作

![image-20220119150203858](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220119150203858.png)

而$M^{(I)}$可看作

![image-20220119150255209](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220119150255209.png)

因此可以得到

![image-20220119150337217](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220119150337217.png)

即**一个卷积核的等效FC核是对单位矩阵进行卷积后再适当变换的结果**。

- RepMLP Block

![image-20220119143705041](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220119143705041.png)

1、Global Perception模块通过pooling和两个FC，从而得到一个编码全局信息的向量

2、Channel Perception模块中包含一个FC层，在此处假设o = c。传统FC有$(chw)^2$个参数，这显然不可接受。我们的解决方法是让多个通道共享一组参数，并且**禁止不同通道间的信息流**。设共有s组参数，那么每$c/s$个通道共享一组参数，此时共有$s\times (hw)^2$个参数。实际操作时，先将输入（n,c,h,w）变换为（nc/s,s,h,w），在得到输出后再反变换即可。

在实践中，Pytorch等框架并不支持参数共享，因此可以使用1x1卷积来代替：

先将$V^{(in)}(\frac{nc}{s},shw)$重塑为为$(\frac{nc}{s},shw,1,1)$；之后再对其进行s组的1x1卷积，最后将输出变换为$(\frac{nc}{s},s,h,w)$

3、Local Perception模块通过分组卷积提取局部特征，其中卷积核参数$F\in R^{s\times 1\times k \times k}$

- 合并层

很容易将卷积层与BN合并：

![image-20220119153948305](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220119153948305.png)

![image-20220119153930951](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220119153930951.png)

而FC3与BN可以合并为$W'\in R^{shw\times hw},b'\in R^{shw}$

之后只需要将每一个卷积核利用之前的变换公式变换后再加到W‘矩阵中即可。

![image-20220119184709706](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220119184709706.png)

代码实现思路很清晰，核心代码就是repmlpnet.py
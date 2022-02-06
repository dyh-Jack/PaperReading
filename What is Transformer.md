## What is Transformer

### 1 一切从Self-attention开始

#### **1.1 处理Sequence数据的模型：**

Transformer是一个Sequence to Sequence model，特别之处在于它大量用到了self-attention。

要处理一个Sequence，最常想到的就是使用RNN，它的输入是一串vector sequence，输出是另一串vector sequence，如下图1左所示。

如果假设是一个single directional的RNN，那当输出 ![[公式]](https://www.zhihu.com/equation?tex=b_4) 时，默认 ![[公式]](https://www.zhihu.com/equation?tex=a_1%2Ca_2%2Ca_3%2Ca_4) 都已经看过了。如果假设是一个bi-directional的RNN，那当输出 ![[公式]](https://www.zhihu.com/equation?tex=b_%7B%E4%BB%BB%E6%84%8F%7D) 时，默认 ![[公式]](https://www.zhihu.com/equation?tex=a_1%2Ca_2%2Ca_3%2Ca_4) 都已经看过了。RNN非常擅长于处理input是一个sequence的状况。

那RNN有什么样的问题呢？它的问题就在于：RNN很不容易并行化 (hard to parallel)。

为什么说RNN很不容易并行化呢？假设在single directional的RNN的情形下，你今天要算出 ![[公式]](https://www.zhihu.com/equation?tex=b_4) ，就必须要先看 ![[公式]](https://www.zhihu.com/equation?tex=a_1) 再看 ![[公式]](https://www.zhihu.com/equation?tex=a_2) 再看 ![[公式]](https://www.zhihu.com/equation?tex=a_3) 再看 ![[公式]](https://www.zhihu.com/equation?tex=a_4) ，所以这个过程很难平行处理。

所以今天就有人提出把CNN拿来取代RNN，如下图1右所示。其中，橘色的三角形表示一个filter，每次扫过3个向量 ![[公式]](https://www.zhihu.com/equation?tex=a) ，扫过一轮以后，就输出了一排结果，使用橘色的小圆点表示。

这是第一个橘色的filter的过程，还有其他的filter，比如图2中的黄色的filter，它经历着与橘色的filter相似的过程，又输出一排结果，使用黄色的小圆点表示。

![img](https://pic2.zhimg.com/80/v2-7a6a6f0977b06b3372b129a09a3ccb31_1440w.jpg)图1：处理Sequence数据的模型

![img](https://pic2.zhimg.com/80/v2-cabda788832922a8f141542a334ccb61_1440w.jpg)**图2：处理Sequence数据的模型**

所以，用CNN，你确实也可以做到跟RNN的输入输出类似的关系，也可以做到输入是一个sequence，输出是另外一个sequence。

但是，表面上CNN和RNN可以做到相同的输入和输出，但是CNN只能考虑非常有限的内容。比如在我们右侧的图中CNN的filter只考虑了3个vector，不像RNN可以考虑之前的所有vector。但是CNN也不是没有办法考虑很长时间的dependency的，你只需要堆叠filter，多堆叠几层，上层的filter就可以考虑比较多的资讯，比如，第二层的filter (蓝色的三角形)看了6个vector，所以，只要叠很多层，就能够看很长时间的资讯。

而CNN的一个好处是：它是可以并行化的 (can parallel)，不需要等待红色的filter算完，再算黄色的filter。但是必须要叠很多层filter，才可以看到长时的资讯。所以今天有一个想法：self-attention，如下图3所示，目的是使用self-attention layer取代RNN所做的事情。

![img](https://pic2.zhimg.com/80/v2-e3ef96ccae817226577ee7a3c28fa16d_1440w.jpg)**图3：You can try to replace any thing that has been done by RNN with self attention**

**所以重点是：我们有一种新的layer，叫self-attention，它的输入和输出和RNN是一模一样的，输入一个sequence，输出一个sequence，它的每一个输出 ![[公式]](https://www.zhihu.com/equation?tex=b_1-b_4) 都看过了整个的输入sequence，这一点与bi-directional RNN相同。但是神奇的地方是：它的每一个输出 ![[公式]](https://www.zhihu.com/equation?tex=b_1-b_4)可以并行化计算。**

#### **1.2 Self-attention：**

**那么self-attention具体是怎么做的呢？**

![img](https://pic3.zhimg.com/80/v2-8537e0996a586b7c37d5e345b6c4402a_1440w.jpg)图4：self-attention具体是怎么做的？

首先假设我们的input是图4的 ![[公式]](https://www.zhihu.com/equation?tex=x_1-x_4) ，是一个sequence，每一个input (vector)先乘上一个矩阵 ![[公式]](https://www.zhihu.com/equation?tex=W) 得到embedding，即向量 ![[公式]](https://www.zhihu.com/equation?tex=a_1-a_4) 。接着这个embedding进入self-attention层，每一个向量 ![[公式]](https://www.zhihu.com/equation?tex=a_1-a_4) 分别乘上3个不同的transformation matrix ![[公式]](https://www.zhihu.com/equation?tex=W_q%2CW_k%2CW_v) ，以向量 ![[公式]](https://www.zhihu.com/equation?tex=a_1) 为例，分别得到3个不同的向量 ![[公式]](https://www.zhihu.com/equation?tex=q_1%2Ck_1%2Cv_1) 。

![img](https://pic4.zhimg.com/80/v2-197b4f81d688e4bc40843fbe41c96787_1440w.jpg)图5：self-attention具体是怎么做的？

接下来使用每个query ![[公式]](https://www.zhihu.com/equation?tex=q) 去对每个key ![[公式]](https://www.zhihu.com/equation?tex=k) 做attention，attention就是匹配这2个向量有多接近，比如我现在要对 ![[公式]](https://www.zhihu.com/equation?tex=q%5E1) 和 ![[公式]](https://www.zhihu.com/equation?tex=k%5E1) 做attention，我就可以把这2个向量做**scaled inner product**，得到 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_%7B1%2C1%7D) 。接下来你再拿 ![[公式]](https://www.zhihu.com/equation?tex=q%5E1) 和 ![[公式]](https://www.zhihu.com/equation?tex=k%5E2) 做attention，得到 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_%7B1%2C2%7D) ，你再拿 ![[公式]](https://www.zhihu.com/equation?tex=q%5E1) 和 ![[公式]](https://www.zhihu.com/equation?tex=k%5E3) 做attention，得到 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_%7B1%2C3%7D) ，你再拿 ![[公式]](https://www.zhihu.com/equation?tex=q%5E1) 和 ![[公式]](https://www.zhihu.com/equation?tex=k%5E4) 做attention，得到 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_%7B1%2C4%7D) 。那这个scaled inner product具体是怎么计算的呢？

![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_%7B1%2Ci%7D%3Dq%5E1%5Ccdot+k%5Ei%2F%5Csqrt%7Bd%7D+%5Ctag%7B1%7D)

式中， ![[公式]](https://www.zhihu.com/equation?tex=d) 是 ![[公式]](https://www.zhihu.com/equation?tex=q) 跟 ![[公式]](https://www.zhihu.com/equation?tex=k) 的维度。因为 ![[公式]](https://www.zhihu.com/equation?tex=q%5Ccdot+k) 的数值会随着dimension的增大而增大，所以要除以 ![[公式]](https://www.zhihu.com/equation?tex=%5Csqrt%7B%5Ctext%7Bdimension%7D%7D) 的值，相当于归一化的效果。

接下来要做的事如图6所示，把计算得到的所有 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_%7B1%2Ci%7D) 值取 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bsoftmax%7D) 操作。

![img](https://pic2.zhimg.com/80/v2-58f7bf32a29535b57205ac2dab557be1_1440w.jpg)图6：self-attention具体是怎么做的？

取完 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bsoftmax%7D) 操作以后，我们得到了 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat+%5Calpha_%7B1%2Ci%7D) ，我们用它和所有的 ![[公式]](https://www.zhihu.com/equation?tex=v%5Ei) 值进行相乘。具体来讲，把 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat+%5Calpha_%7B1%2C1%7D) 乘上 ![[公式]](https://www.zhihu.com/equation?tex=v%5E1) ，把 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat+%5Calpha_%7B1%2C2%7D) 乘上 ![[公式]](https://www.zhihu.com/equation?tex=v%5E2) ，把 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat+%5Calpha_%7B1%2C3%7D) 乘上 ![[公式]](https://www.zhihu.com/equation?tex=v%5E3) ，把 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat+%5Calpha_%7B1%2C4%7D) 乘上 ![[公式]](https://www.zhihu.com/equation?tex=v%5E4) ，把结果通通加起来得到 ![[公式]](https://www.zhihu.com/equation?tex=b%5E1) ，所以，今天在产生 ![[公式]](https://www.zhihu.com/equation?tex=b%5E1) 的过程中用了整个sequence的资讯 (Considering the whole sequence)。如果要考虑local的information，则只需要学习出相应的 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat+%5Calpha_%7B1%2Ci%7D%3D0) ， ![[公式]](https://www.zhihu.com/equation?tex=b%5E1) 就不再带有那个对应分支的信息了；如果要考虑global的information，则只需要学习出相应的 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat+%5Calpha_%7B1%2Ci%7D%5Cne0) ， ![[公式]](https://www.zhihu.com/equation?tex=b%5E1) 就带有全部的对应分支的信息了。

![img](https://pic3.zhimg.com/80/v2-b7e1ffade85d4dbe3350f23e6854c272_1440w.jpg)图7：self-attention具体是怎么做的？

同样的方法，也可以计算出 ![[公式]](https://www.zhihu.com/equation?tex=b%5E2%2Cb%5E3%2Cb%5E4) ，如下图8所示， ![[公式]](https://www.zhihu.com/equation?tex=b%5E2) 就是拿query ![[公式]](https://www.zhihu.com/equation?tex=q%5E2)去对其他的 ![[公式]](https://www.zhihu.com/equation?tex=k) 做attention，得到 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat+%5Calpha_%7B2%2Ci%7D) ，再与value值 ![[公式]](https://www.zhihu.com/equation?tex=v%5Ei) 相乘取weighted sum得到的。

![img](https://pic2.zhimg.com/80/v2-f7b03e1979c6ccd1dab4b579654c8cd5_1440w.jpg)图8：self-attention具体是怎么做的？

经过了以上一连串计算，self-attention layer做的事情跟RNN是一样的，只是它可以并行的得到layer输出的结果，如图9所示。现在我们要用矩阵表示上述的计算过程。

![img](https://pic2.zhimg.com/80/v2-67bc90b683b40488e922dcd5abcaa089_1440w.jpg)图9：self-attention的效果

首先输入的embedding是 ![[公式]](https://www.zhihu.com/equation?tex=I%3D%5Ba%5E1%2Ca%5E2%2Ca%5E3%2Ca%5E4%5D) ，然后用 ![[公式]](https://www.zhihu.com/equation?tex=I) 乘以transformation matrix ![[公式]](https://www.zhihu.com/equation?tex=W%5Eq) 得到 ![[公式]](https://www.zhihu.com/equation?tex=Q%3D%5Bq%5E1%2Cq%5E2%2Cq%5E3%2Cq%5E4%5D) ，它的每一列代表着一个vector ![[公式]](https://www.zhihu.com/equation?tex=q) 。同理，用 ![[公式]](https://www.zhihu.com/equation?tex=I) 乘以transformation matrix ![[公式]](https://www.zhihu.com/equation?tex=W%5Ek) 得到 ![[公式]](https://www.zhihu.com/equation?tex=K%3D%5Bk%5E1%2Ck%5E2%2Ck%5E3%2Ck%5E4%5D) ，它的每一列代表着一个vector ![[公式]](https://www.zhihu.com/equation?tex=k) 。用 ![[公式]](https://www.zhihu.com/equation?tex=I) 乘以transformation matrix ![[公式]](https://www.zhihu.com/equation?tex=W%5Ev) 得到 ![[公式]](https://www.zhihu.com/equation?tex=V%3D%5Bv%5E1%2Cv%5E2%2Cv%5E3%2Cv%5E4%5D) ，它的每一列代表着一个vector ![[公式]](https://www.zhihu.com/equation?tex=v) 。

![img](https://pic2.zhimg.com/80/v2-b081f7cbc5ecd2471567426e696bde15_1440w.jpg)图10：self-attention的矩阵计算过程

接下来是 ![[公式]](https://www.zhihu.com/equation?tex=k) 与 ![[公式]](https://www.zhihu.com/equation?tex=q) 的attention过程，我们可以把vector ![[公式]](https://www.zhihu.com/equation?tex=k) 横过来变成行向量，与列向量 ![[公式]](https://www.zhihu.com/equation?tex=q) 做内积，这里省略了 ![[公式]](https://www.zhihu.com/equation?tex=%5Csqrt%7Bd%7D) 。这样， ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 就成为了 ![[公式]](https://www.zhihu.com/equation?tex=4%5Ctimes4) 的矩阵，它由4个行向量拼成的矩阵和4个列向量拼成的矩阵做内积得到，如图11所示。

在得到 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat+A) 以后，如上文所述，要得到 ![[公式]](https://www.zhihu.com/equation?tex=b%5E1)， 就要使用 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat+%5Calpha_%7B1%2Ci%7D) 分别与 ![[公式]](https://www.zhihu.com/equation?tex=v%5Ei) 相乘再求和得到，所以 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat+A) 要再左乘 ![[公式]](https://www.zhihu.com/equation?tex=V) 矩阵。

![img](https://pic3.zhimg.com/80/v2-6cc342a83d25ac76b767b5bbf27d9d6e_1440w.jpg)

![img](https://pic2.zhimg.com/80/v2-52a5e6b928dc44db73f85001b2d1133d_1440w.jpg)

![img](https://pic4.zhimg.com/80/v2-1b7d30f098f02488c48c3601f8e13033_1440w.jpg)图11：self-attention的矩阵计算过程

到这里你会发现这个过程可以被表示为，如图12所示：输入矩阵 ![[公式]](https://www.zhihu.com/equation?tex=I%5Cin+R+%28d%2CN%29) 分别乘上3个不同的矩阵 ![[公式]](https://www.zhihu.com/equation?tex=W_q%2CW_k%2CW_v+%5Cin+R+%28d%2Cd%29) 得到3个中间矩阵 ![[公式]](https://www.zhihu.com/equation?tex=Q%2CK%2CV%5Cin+R+%28d%2CN%29) 。它们的维度是相同的。把 ![[公式]](https://www.zhihu.com/equation?tex=K) 转置之后与 ![[公式]](https://www.zhihu.com/equation?tex=Q) 相乘得到Attention矩阵 ![[公式]](https://www.zhihu.com/equation?tex=A%5Cin+R+%28N%2CN%29) ，代表每一个位置两两之间的attention。再将它取 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bsoftmax%7D) 操作得到 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat+A%5Cin+R+%28N%2CN%29) ，最后将它乘以 ![[公式]](https://www.zhihu.com/equation?tex=V) 矩阵得到输出vector ![[公式]](https://www.zhihu.com/equation?tex=O%5Cin+R+%28d%2CN%29) 。

![[公式]](https://www.zhihu.com/equation?tex=%5Chat+A%3D%5Ctext%7Bsoftmax%7D%28A%29%3DK%5ET%5Ccdot+Q+%5Ctag%7B2%7D)

![[公式]](https://www.zhihu.com/equation?tex=O%3DV%5Ccdot%5Chat+A%5Ctag%7B3%7D)

![img](https://pic2.zhimg.com/80/v2-8628bf2c2bb9a7ee2c4a0fb870ab32b9_1440w.jpg)图12：self-attention就是一堆矩阵乘法，可以实现GPU加速

#### **1.3 Multi-head Self-attention：**

还有一种multi-head的self-attention，以2个head的情况为例：由 ![[公式]](https://www.zhihu.com/equation?tex=a%5Ei) 生成的 ![[公式]](https://www.zhihu.com/equation?tex=q%5Ei) 进一步乘以2个转移矩阵变为 ![[公式]](https://www.zhihu.com/equation?tex=q%5E%7Bi%2C1%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=q%5E%7Bi%2C2%7D) ，同理由 ![[公式]](https://www.zhihu.com/equation?tex=a%5Ei) 生成的 ![[公式]](https://www.zhihu.com/equation?tex=k%5Ei) 进一步乘以2个转移矩阵变为 ![[公式]](https://www.zhihu.com/equation?tex=k%5E%7Bi%2C1%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=k%5E%7Bi%2C2%7D) ，由 ![[公式]](https://www.zhihu.com/equation?tex=a%5Ei) 生成的 ![[公式]](https://www.zhihu.com/equation?tex=v%5Ei) 进一步乘以2个转移矩阵变为 ![[公式]](https://www.zhihu.com/equation?tex=v%5E%7Bi%2C1%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=v%5E%7Bi%2C2%7D) 。接下来 ![[公式]](https://www.zhihu.com/equation?tex=q%5E%7Bi%2C1%7D) 再与 ![[公式]](https://www.zhihu.com/equation?tex=k%5E%7Bi%2C1%7D) 做attention，得到weighted sum的权重 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) ，再与 ![[公式]](https://www.zhihu.com/equation?tex=v%5E%7Bi%2C1%7D) 做weighted sum得到最终的 ![[公式]](https://www.zhihu.com/equation?tex=b%5E%7Bi%2C1%7D%28i%3D1%2C2%2C...%2CN%29) 。同理得到 ![[公式]](https://www.zhihu.com/equation?tex=b%5E%7Bi%2C2%7D%28i%3D1%2C2%2C...%2CN%29) 。现在我们有了 ![[公式]](https://www.zhihu.com/equation?tex=b%5E%7Bi%2C1%7D%28i%3D1%2C2%2C...%2CN%29%5Cin+R%28d%2C1%29) 和 ![[公式]](https://www.zhihu.com/equation?tex=b%5E%7Bi%2C2%7D%28i%3D1%2C2%2C...%2CN%29%5Cin+R%28d%2C1%29) ，可以把它们concat起来，再通过一个transformation matrix调整维度，使之与刚才的 ![[公式]](https://www.zhihu.com/equation?tex=b%5E%7Bi%7D%28i%3D1%2C2%2C...%2CN%29%5Cin+R%28d%2C1%29) 维度一致(这步如图13所示)。

![img](https://pic1.zhimg.com/80/v2-688516477ad57f01a4abe5fd1a36e510_1440w.jpg)

![img](https://pic3.zhimg.com/80/v2-b0891e9352874c9eee469372b85ecbe2_1440w.jpg)图13：multi-head self-attention

![img](https://pic1.zhimg.com/80/v2-df5d332304c2fd217705f210edd18bf4_1440w.jpg)

图13：调整b的维度

从下图14可以看到 Multi-Head Attention 包含多个 Self-Attention 层，首先将输入 ![[公式]](https://www.zhihu.com/equation?tex=X) 分别传递到 2个不同的 Self-Attention 中，计算得到 2 个输出结果。得到2个输出矩阵之后，Multi-Head Attention 将它们拼接在一起 (Concat)，然后传入一个Linear层，得到 Multi-Head Attention 最终的输出 ![[公式]](https://www.zhihu.com/equation?tex=Z) 。可以看到 Multi-Head Attention 输出的矩阵 ![[公式]](https://www.zhihu.com/equation?tex=Z) 与其输入的矩阵 ![[公式]](https://www.zhihu.com/equation?tex=X) 的维度是一样的。

![img](https://pic2.zhimg.com/80/v2-f784c73ae6eb34a00108b64e3db394fd_1440w.jpg)

图14：multi-head self-attention

这里有一组Multi-head Self-attention的解果，其中绿色部分是一组query和key，红色部分是另外一组query和key，可以发现绿色部分其实更关注global的信息，而红色部分其实更关注local的信息。

![img](https://pic3.zhimg.com/80/v2-6b6c906cfca399506d324cac3292b04a_1440w.jpg)图15：Multi-head Self-attention的不同head分别关注了global和local的讯息

#### **1.4 Positional Encoding：**

以上是multi-head self-attention的原理，但是还有一个问题是：现在的self-attention中没有位置的信息，一个单词向量的“近在咫尺”位置的单词向量和“远在天涯”位置的单词向量效果是一样的，没有表示位置的信息(No position information in self attention)。所以你输入"A打了B"或者"B打了A"的效果其实是一样的，因为并没有考虑位置的信息。所以在self-attention原来的paper中，作者为了解决这个问题所做的事情是如下图16所示：

![img](https://pic3.zhimg.com/80/v2-b8886621fc841085300f5bb21de26f0e_1440w.jpg)

![img](https://pic4.zhimg.com/80/v2-7814595d02ef37cb762b3ef998fae267_1440w.jpg)图16：self-attention中的位置编码

具体的做法是：给每一个位置规定一个表示位置信息的向量 ![[公式]](https://www.zhihu.com/equation?tex=e%5Ei) ，让它与 ![[公式]](https://www.zhihu.com/equation?tex=a%5Ei) 加在一起之后作为新的 ![[公式]](https://www.zhihu.com/equation?tex=a%5Ei) 参与后面的运算过程，但是这个向量 ![[公式]](https://www.zhihu.com/equation?tex=e%5Ei) 是由人工设定的，而不是神经网络学习出来的。每一个位置都有一个不同的 ![[公式]](https://www.zhihu.com/equation?tex=e%5Ei) 。

那到这里一个自然而然的问题是：**为什么是 ![[公式]](https://www.zhihu.com/equation?tex=e%5Ei) 与 ![[公式]](https://www.zhihu.com/equation?tex=a%5Ei) 相加？为什么不是concatenate？加起来以后，原来表示位置的资讯不就混到 ![[公式]](https://www.zhihu.com/equation?tex=a%5Ei) 里面去了吗？不就很难被找到了吗？**

**这里提供一种解答这个问题的思路：**

如图15所示，我们先给每一个位置的 ![[公式]](https://www.zhihu.com/equation?tex=x%5Ei%5Cin+R%28d%2C1%29) append一个one-hot编码的向量 ![[公式]](https://www.zhihu.com/equation?tex=p%5Ei%5Cin+R%28N%2C1%29) ，得到一个新的输入向量 ![[公式]](https://www.zhihu.com/equation?tex=x_p%5Ei%5Cin+R%28d%2BN%2C1%29) ，这个向量作为新的输入，乘以一个transformation matrix ![[公式]](https://www.zhihu.com/equation?tex=W%3D%5BW%5EI%2CW%5EP%5D%5Cin+R%28d%2Cd%2BN%29) 。那么：

![[公式]](https://www.zhihu.com/equation?tex=W%5Ccdot+x_p%5Ei%3D%5BW%5EI%2CW%5EP%5D%5Ccdot%5Cbegin%7Bbmatrix%7Dx%5Ei%5C%5Cp%5Ei+%5Cend%7Bbmatrix%7D%3DW%5EI%5Ccdot+x%5Ei%2BW%5EP%5Ccdot+p%5Ei%3Da%5Ei%2Be%5Ei+%5Ctag%7B4%7D)

**所以，![[公式]](https://www.zhihu.com/equation?tex=e%5Ei) 与 ![[公式]](https://www.zhihu.com/equation?tex=a%5Ei) 相加就等同于把原来的输入 ![[公式]](https://www.zhihu.com/equation?tex=x%5Ei) concat一个表示位置的独热编码 ![[公式]](https://www.zhihu.com/equation?tex=p%5Ei) ，再做transformation。**

**这个与位置编码乘起来的矩阵** ![[公式]](https://www.zhihu.com/equation?tex=W%5EP) 是手工设计的，如图17所示 (黑色框代表一个位置的编码)。

![img](https://pic4.zhimg.com/80/v2-8b7cf3525520292bdfa159463d9717db_1440w.jpg)

图17：与位置编码乘起来的转移矩阵WP

Transformer 中除了单词的 Embedding，还需要使用位置 Embedding 表示单词出现在句子中的位置。因为 Transformer 不采用 RNN 的结构，而是使用全局信息，不能利用单词的顺序信息，而这部分信息对于 NLP 来说非常重要。所以 Transformer 中使用位置 Embedding 保存单词在序列中的相对或绝对位置。

位置 Embedding 用 PE表示，PE 的维度与单词 Embedding 是一样的。PE 可以通过训练得到，也可以使用某种公式计算得到。在 Transformer 中采用了后者，计算公式如下：



![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7DPE_%7B%28pos%2C+2i%29%7D+%3D+sin%28pos%2F10000%5E%7B2i%2Fd_%7Bmodel%7D%7D%29+%5C%5C+PE_%7B%28pos%2C+2i%2B1%29%7D+%3D+cos%28pos%2F10000%5E%7B2i%2Fd_%7Bmodel%7D%7D%29++%5Cend%7Balign%7D%5Ctag%7B5%7D)

式中， ![[公式]](https://www.zhihu.com/equation?tex=pos) 表示token在sequence中的位置，例如第一个token "我" 的 ![[公式]](https://www.zhihu.com/equation?tex=pos%3D0) 。

![[公式]](https://www.zhihu.com/equation?tex=i) ，或者准确意义上是 ![[公式]](https://www.zhihu.com/equation?tex=2i) 和 ![[公式]](https://www.zhihu.com/equation?tex=2i%2B1) 表示了Positional Encoding的维度，![[公式]](https://www.zhihu.com/equation?tex=i) 的取值范围是： ![[公式]](https://www.zhihu.com/equation?tex=%5Cleft%5B+0%2C%5Cldots+%2C%7B%7B%7Bd%7D_%7Bmodel%7D%7D%7D%2F%7B2%7D%5C%3B+%5Cright%29) 。所以当 ![[公式]](https://www.zhihu.com/equation?tex=pos) 为1时，对应的Positional Encoding可以写成：

![[公式]](https://www.zhihu.com/equation?tex=PE%5Cleft%28+1+%5Cright%29%3D%5Cleft%5B+%5Csin+%5Cleft%28+%7B1%7D%2F%7B%7B%7B10000%7D%5E%7B%7B0%7D%2F%7B512%7D%5C%3B%7D%7D%7D%5C%3B+%5Cright%29%2C%5Ccos+%5Cleft%28+%7B1%7D%2F%7B%7B%7B10000%7D%5E%7B%7B0%7D%2F%7B512%7D%5C%3B%7D%7D%7D%5C%3B+%5Cright%29%2C%5Csin+%5Cleft%28+%7B1%7D%2F%7B%7B%7B10000%7D%5E%7B%7B2%7D%2F%7B512%7D%5C%3B%7D%7D%7D%5C%3B+%5Cright%29%2C%5Ccos+%5Cleft%28+%7B1%7D%2F%7B%7B%7B10000%7D%5E%7B%7B2%7D%2F%7B512%7D%5C%3B%7D%7D%7D%5C%3B+%5Cright%29%2C%5Cldots++%5Cright%5D)

式中， ![[公式]](https://www.zhihu.com/equation?tex=%7B%7Bd%7D_%7Bmodel%7D%7D%3D512)。底数是10000。为什么要使用10000呢，这个就类似于玄学了，原论文中完全没有提啊，这里不得不说说论文的readability的问题，即便是很多高引的文章，最基本的内容都讨论不清楚，所以才出现像上面提问里的讨论，说实话这些论文还远远没有做到easy to follow。这里我给出一个假想：![[公式]](https://www.zhihu.com/equation?tex=%7B%7B10000%7D%5E%7B%7B1%7D%2F%7B512%7D%7D%7D)是一个比较接近1的数（1.018），如果用100000，则是1.023。这里只是猜想一下，其实大家应该完全可以使用另一个底数。

这个式子的好处是：

- 每个位置有一个唯一的positional encoding。
- 使 ![[公式]](https://www.zhihu.com/equation?tex=PE) 能够适应比训练集里面所有句子更长的句子，假设训练集里面最长的句子是有 20 个单词，突然来了一个长度为 21 的句子，则使用公式计算的方法可以计算出第 21 位的 Embedding。
- 可以让模型容易地计算出相对位置，对于固定长度的间距 ![[公式]](https://www.zhihu.com/equation?tex=k) ，任意位置的 ![[公式]](https://www.zhihu.com/equation?tex=PE_%7Bpos%2Bk%7D) 都可以被 ![[公式]](https://www.zhihu.com/equation?tex=PE_%7Bpos%7D) 的线性函数表示，因为三角函数特性：

![[公式]](https://www.zhihu.com/equation?tex=cos%28%5Calpha%2B%5Cbeta%29+%3D+cos%28%5Calpha%29cos%28%5Cbeta%29-sin%28%5Calpha%29sin%28%5Cbeta%29+%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=sin%28%5Calpha%2B%5Cbeta%29+%3D+sin%28%5Calpha%29cos%28%5Cbeta%29+%2B+cos%28%5Calpha%29sins%28%5Cbeta%29+%5C%5C)

除了以上的固定位置编码以外，还有其他的很多表示方法：

比如下图18a就是sin-cos的固定位置编码。图b就是可学习的位置编码。图c和d分别FLOATER和RNN模型学习的位置编码。

![img](https://pic3.zhimg.com/80/v2-4ef2648c2bebe2621c0c03001c0e1b92_1440w.jpg)图18：其他的很多位置编码方法

接下来我们看看self-attention在sequence2sequence model里面是怎么使用的，我们可以把Encoder-Decoder中的RNN用self-attention取代掉。

![img](https://pic4.zhimg.com/80/v2-287ebca58558012f9459f3f1d5bc3827_1440w.jpg)图18：Seq2seq with Attention

在self-attention的最后一部分我们来对比下self-attention和CNN的关系。如图19，今天在使用self-attention去处理一张图片的时候，1的那个pixel产生query，其他的各个pixel产生key。在做inner-product的时候，考虑的不是一个小的范围，而是一整张图片。

但是在做CNN的时候是只考虑感受野红框里面的资讯，而不是图片的全局信息。所以CNN可以看作是一种简化版本的self-attention。

或者可以反过来说，self-attention是一种复杂化的CNN，在做CNN的时候是只考虑感受野红框里面的资讯，而感受野的范围和大小是由人决定的。但是self-attention由attention找到相关的pixel，就好像是感受野的范围和大小是自动被学出来的，所以CNN可以看做是self-attention的特例，如图20所示。

![img](https://pic3.zhimg.com/80/v2-f28a8b0295863ab78d92a281ae55fce2_1440w.jpg)

图19：CNN考虑感受野范围，而self-attention考虑的不是一个小的范围，而是一整张图片

![img](https://pic4.zhimg.com/80/v2-f268035371aa22a350a317fc237a04f7_1440w.jpg)

图20：CNN可以看做是self-attention的特例

既然self-attention是更广义的CNN，则这个模型更加flexible。而我们认为，一个模型越flexible，训练它所需要的数据量就越多，所以在训练self-attention模型时就需要更多的数据，这一点在下面介绍的论文 ViT 中有印证，它需要的数据集是有3亿张图片的JFT-300，而如果不使用这么多数据而只使用ImageNet，则性能不如CNN。

### 2 Transformer的实现和代码解读

#### **2.1 Transformer原理分析：**

![img](https://pic4.zhimg.com/80/v2-1719966a223d98ad48f98c2e4d71add7_1440w.jpg)图21：Transformer

> **Encoder：**

这个图21讲的是一个seq2seq的model，左侧为 Encoder block，右侧为 Decoder block。红色圈中的部分为Multi-Head Attention，是由多个Self-Attention组成的，可以看到 Encoder block 包含一个 Multi-Head Attention，而 Decoder block 包含两个 Multi-Head Attention (其中有一个用到 Masked)。Multi-Head Attention 上方还包括一个 Add & Norm 层，Add 表示残差连接 (Residual Connection) 用于防止网络退化，Norm 表示 Layer Normalization，用于对每一层的激活值进行归一化。比如说在Encoder Input处的输入是机器学习，在Decoder Input处的输入是<BOS>，输出是machine。再下一个时刻在Decoder Input处的输入是machine，输出是learning。不断重复知道输出是句点(.)代表翻译结束。

接下来我们看看这个Encoder和Decoder里面分别都做了什么事情，先看左半部分的Encoder：首先输入 ![[公式]](https://www.zhihu.com/equation?tex=X%5Cin+R+%28n_x%2CN%29) 通过一个Input Embedding的转移矩阵 ![[公式]](https://www.zhihu.com/equation?tex=W%5EX%5Cin+R+%28d%2Cn_x%29) 变为了一个张量，即上文所述的 ![[公式]](https://www.zhihu.com/equation?tex=I%5Cin+R+%28d%2CN%29) ，再加上一个表示位置的Positional Encoding ![[公式]](https://www.zhihu.com/equation?tex=E%5Cin+R+%28d%2CN%29) ，得到一个张量，去往后面的操作。

它进入了这个绿色的block，这个绿色的block会重复 ![[公式]](https://www.zhihu.com/equation?tex=N) 次。这个绿色的block里面有什么呢？它的第1层是一个上文讲的multi-head的attention。你现在一个sequence ![[公式]](https://www.zhihu.com/equation?tex=I%5Cin+R+%28d%2CN%29) ，经过一个multi-head的attention，你会得到另外一个sequence ![[公式]](https://www.zhihu.com/equation?tex=O%5Cin+R+%28d%2CN%29) 。

下一个Layer是Add & Norm，这个意思是说：把multi-head的attention的layer的输入 ![[公式]](https://www.zhihu.com/equation?tex=I%5Cin+R+%28d%2CN%29) 和输出 ![[公式]](https://www.zhihu.com/equation?tex=O%5Cin+R+%28d%2CN%29) 进行相加以后，再做Layer Normalization，至于Layer Normalization和我们熟悉的Batch Normalization的区别是什么，请参考图20和21。



![img](https://pic3.zhimg.com/80/v2-53267aa305030eb71376296a6fd14cde_1440w.jpg)图22：不同Normalization方法的对比

其中，Batch Normalization和Layer Normalization的对比可以概括为图22，Batch Normalization强行让一个batch的数据的某个channel的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmu%3D0%2C%5Csigma%3D1) ，而Layer Normalization让一个数据的所有channel的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmu%3D0%2C%5Csigma%3D1) 。

![img](https://pic1.zhimg.com/80/v2-4c13b36ec9a6a2d2f4911d2d9e7122b8_1440w.jpg)

图23：Batch Normalization和Layer Normalization的对比

接着是一个Feed Forward的前馈网络和一个Add & Norm Layer。

所以，这一个绿色的block的前2个Layer操作的表达式为：

![[公式]](https://www.zhihu.com/equation?tex=%5Ccolor%7Bdarkgreen%7D%7BO_1%7D%3D%5Ccolor%7Bgreen%7D%7B%5Ctext%7BLayer+Normalization%7D%7D%28%5Ccolor%7Bteal%7D%7BI%7D%2B%5Ccolor%7Bcrimson%7D%7B%5Ctext%7BMulti-head+Self-Attention%7D%7D%28%5Ccolor%7Bteal%7D%7BI%7D%29%29%5Ctag%7B6%7D)

这一个绿色的block的后2个Layer操作的表达式为：

![[公式]](https://www.zhihu.com/equation?tex=%5Ccolor%7Bdarkgreen%7D%7BO_2%7D%3D%5Ccolor%7Bgreen%7D%7B%5Ctext%7BLayer+Normalization%7D%7D%28%5Ccolor%7Bteal%7D%7BO_1%7D%2B%5Ccolor%7Bcrimson%7D%7B%5Ctext%7BFeed+Forward+Network%7D%7D%28%5Ccolor%7Bteal%7D%7BO_1%7D%29%29%5Ctag%7B7%7D)

![[公式]](https://www.zhihu.com/equation?tex=%5Ccolor%7Bgreen%7D%7B%5Ctext%7BBlock%7D%7D%28%5Ccolor%7Bteal%7D%7BI%7D%29%3D%5Ccolor%7Bgreen%7D%7BO_2%7D+%5Ctag%7B8%7D)

所以Transformer的Encoder的整体操作为：

![[公式]](https://www.zhihu.com/equation?tex=%5Ccolor%7Bpurple%7D%7B%5Ctext%7BEncoder%7D%7D%28%5Ccolor%7Bdarkgreen%7D%7BI%7D%29%3D%5Ccolor%7Bdarkgreen%7D%7B%5Ctext%7BBlock%7D%7D%28...%5Ccolor%7Bdarkgreen%7D%7B%5Ctext%7BBlock%7D%7D%28%5Ccolor%7Bdarkgreen%7D%7B%5Ctext%7BBlock%7D%7D%29%28%5Ccolor%7Bteal%7D%7BI%7D%29%29%5C%5C%5Cquad+N%5C%3Btimes+%5Ctag%7B9%7D+)



> **Decoder：**

现在来看Decoder的部分，输入包括2部分，下方是前一个time step的输出的embedding，即上文所述的 ![[公式]](https://www.zhihu.com/equation?tex=I%5Cin+R+%28d%2CN%29) ，再加上一个表示位置的Positional Encoding ![[公式]](https://www.zhihu.com/equation?tex=E%5Cin+R+%28d%2CN%29) ，得到一个张量，去往后面的操作。它进入了这个绿色的block，这个绿色的block会重复 ![[公式]](https://www.zhihu.com/equation?tex=N) 次。这个绿色的block里面有什么呢？

首先是Masked Multi-Head Self-attention，masked的意思是使attention只会attend on已经产生的sequence，这个很合理，因为还没有产生出来的东西不存在，就无法做attention。

**输出是：**对应 ![[公式]](https://www.zhihu.com/equation?tex=%5Ccolor%7Bcrimson%7D%7Bi%7D) 位置的输出词的概率分布。

**输入是：** ![[公式]](https://www.zhihu.com/equation?tex=%5Ccolor%7Bpurple%7D%7BEncoder%7D) **的输出** 和 **对应** ![[公式]](https://www.zhihu.com/equation?tex=%5Ccolor%7Bcrimson%7D%7Bi-1%7D) **位置decoder的输出**。所以中间的attention不是self-attention，它的Key和Value来自encoder，Query来自上一位置 ![[公式]](https://www.zhihu.com/equation?tex=%5Ccolor%7Bcrimson%7D%7BDecoder%7D) 的输出。

**解码：这里要特别注意一下，编码可以并行计算，一次性全部Encoding出来，但解码不是一次把所有序列解出来的，而是像**RNN **一样一个一个解出来的**，因为**<u>要用上一个位置的输入当作attention的query</u>**。

明确了解码过程之后最上面的图就很好懂了，这里主要的不同就是新加的另外要说一下新加的attention多加了一个mask，因为训练时的output都是Ground Truth，这样可以确保预测第 ![[公式]](https://www.zhihu.com/equation?tex=%5Ccolor%7Bcrimson%7D%7Bi%7D) 个位置时不会接触到未来的信息。

- 包含两个 Multi-Head Attention 层。
- 第一个 Multi-Head Attention 层采用了 Masked 操作。
- 第二个 Multi-Head Attention 层的Key，Value矩阵使用 Encoder 的编码信息矩阵 ![[公式]](https://www.zhihu.com/equation?tex=C) 进行计算，而Query使用上一个 Decoder block 的输出计算。
- 最后有一个 Softmax 层计算下一个翻译单词的概率。

下面详细介绍下Masked Multi-Head Self-attention的具体操作，**Masked在Scale操作之后，softmax操作之前**。

![img](https://pic3.zhimg.com/80/v2-58ac6e864d336abce052cf36d480cfee_1440w.jpg)

图24：Masked在Scale操作之后，softmax操作之前

因为在翻译的过程中是顺序翻译的，即翻译完第 ![[公式]](https://www.zhihu.com/equation?tex=i) 个单词，才可以翻译第 ![[公式]](https://www.zhihu.com/equation?tex=i%2B1) 个单词。通过 Masked 操作可以防止第 ![[公式]](https://www.zhihu.com/equation?tex=i) 个单词知道第 ![[公式]](https://www.zhihu.com/equation?tex=i%2B1) 个单词之后的信息。下面以 "我有一只猫" 翻译成 "I have a cat" 为例，了解一下 Masked 操作。在 Decoder 的时候，是需要根据之前的翻译，求解当前最有可能的翻译，如下图所示。首先根据输入 "<Begin>" 预测出第一个单词为 "I"，然后根据输入 "<Begin> I" 预测下一个单词 "have"。

Decoder 可以在训练的过程中使用 Teacher Forcing **并且并行化训练，即将正确的单词序列 (<Begin> I have a cat) 和对应输出 (I have a cat <end>) 传递到 Decoder。那么在预测第** ![[公式]](https://www.zhihu.com/equation?tex=i) **个输出时，就要将第** ![[公式]](https://www.zhihu.com/equation?tex=i%2B1) **之后的单词掩盖住，**注意 Mask 操作是在 Self-Attention 的 Softmax 之前使用的，下面用 0 1 2 3 4 5 分别表示 "<Begin> I have a cat <end>"。

![img](https://pic1.zhimg.com/80/v2-20d6a9f4b3cc8cbae05778816d1af414_1440w.jpg)图25：Decoder过程

注意这里transformer模型训练和测试的方法不同：

**测试时：**

1. 输入<Begin>，解码器输出 I 。
2. 输入前面已经解码的<Begin>和 I，解码器输出have。
3. 输入已经解码的<Begin>，I, have, a, cat，解码器输出解码结束标志位<end>，每次解码都会利用前面已经解码输出的所有单词嵌入信息。

**训练时：**

**不采用上述类似RNN的方法**一个一个目标单词嵌入向量顺序输入训练，想采用**类似编码器中的矩阵并行算法，一步就把所有目标单词预测出来**。要实现这个功能就可以参考编码器的操作，把目标单词嵌入向量组成矩阵一次输入即可。即：**并行化训练。**

但是在解码have时候，不能利用到后面单词a和cat的目标单词嵌入向量信息，否则这就是作弊(测试时候不可能能未卜先知)。为此引入mask。具体是：在解码器中，self-attention层只被允许处理输出序列中更靠前的那些位置，在softmax步骤前，它会把后面的位置给隐去。

**Masked Multi-Head Self-attention的具体操作**如图26所示。

**Step1：**输入矩阵包含 "<Begin> I have a cat" (0, 1, 2, 3, 4) 五个单词的表示向量，Mask是一个 5×5 的矩阵。在Mask可以发现单词 0 只能使用单词 0 的信息，而单词 1 可以使用单词 0, 1 的信息，即只能使用之前的信息。输入矩阵 ![[公式]](https://www.zhihu.com/equation?tex=X%5Cin+R_%7BN%2Cd_x%7D) 经过transformation matrix变为3个矩阵：Query ![[公式]](https://www.zhihu.com/equation?tex=Q%5Cin+R_%7BN%2Cd%7D) ，Key ![[公式]](https://www.zhihu.com/equation?tex=K%5Cin+R_%7BN%2Cd%7D) 和Value ![[公式]](https://www.zhihu.com/equation?tex=V%5Cin+R_%7BN%2Cd%7D) 。

**Step2：** ![[公式]](https://www.zhihu.com/equation?tex=Q%5ET%5Ccdot+K) 得到 Attention矩阵 ![[公式]](https://www.zhihu.com/equation?tex=A%5Cin+R_%7BN%2CN%7D) ，此时先不急于做softmax的操作，而是先于一个 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BMask%7D%5Cin+R_%7BN%2CN%7D) 矩阵相乘，使得attention矩阵的有些位置 归0，得到Masked Attention矩阵 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BMask+Attention%7D%5Cin+R_%7BN%2CN%7D) 。 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BMask%7D%5Cin+R_%7BN%2CN%7D) 矩阵是个下三角矩阵，为什么这样设计？是因为想在计算 ![[公式]](https://www.zhihu.com/equation?tex=Z) 矩阵的某一行时，只考虑它前面token的作用。即：在计算 ![[公式]](https://www.zhihu.com/equation?tex=Z) 的第一行时，刻意地把 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BAttention%7D) 矩阵第一行的后面几个元素屏蔽掉，只考虑 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BAttention%7D_%7B0%2C0%7D) 。在产生have这个单词时，只考虑 I，不考虑之后的have a cat，即只会attend on已经产生的sequence，这个很合理，因为还没有产生出来的东西不存在，就无法做attention。

**Step3：**Masked Attention矩阵进行 Softmax，每一行的和都为 1。但是单词 0 在单词 1, 2, 3, 4 上的 attention score 都为 0。得到的结果再与 ![[公式]](https://www.zhihu.com/equation?tex=V) 矩阵相乘得到最终的self-attention层的输出结果 ![[公式]](https://www.zhihu.com/equation?tex=Z_1%5Cin+R_%7BN%2Cd%7D) 。

**Step4：** ![[公式]](https://www.zhihu.com/equation?tex=Z_1%5Cin+R_%7BN%2Cd%7D) 只是某一个head的结果，将多个head的结果concat在一起之后再最后进行Linear Transformation得到最终的Masked Multi-Head Self-attention的输出结果 ![[公式]](https://www.zhihu.com/equation?tex=Z%5Cin+R_%7BN%2Cd%7D) 。

![img](https://pic4.zhimg.com/80/v2-b32b3c632a20f8daf12103dd05587fd7_1440w.jpg)图26：Masked Multi-Head Self-attention的具体操作

第1个**Masked Multi-Head Self-attention**的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BQuery%2C+Key%2C+Value%7D) 均来自Output Embedding。

第2个**Multi-Head Self-attention**的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BQuery%7D) 来自第1个Self-attention layer的输出， ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BKey%2C+Value%7D) 来自Encoder的输出。

**为什么这么设计？**这里提供一种个人的理解：

![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BKey%2C+Value%7D) 来自Transformer Encoder的输出，所以可以看做**句子(Sequence)/图片(image)**的**内容信息(content，比如句意是："我有一只猫"，图片内容是："有几辆车，几个人等等")**。

![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BQuery%7D) 表达了一种诉求：希望得到什么，可以看做**引导信息(guide)**。

通过Multi-Head Self-attention结合在一起的过程就相当于是**把我们需要的内容信息指导表达出来**。

Decoder的最后是Softmax 预测输出单词。因为 Mask 的存在，使得单词 0 的输出 ![[公式]](https://www.zhihu.com/equation?tex=Z%280%2C%29) 只包含单词 0 的信息。Softmax 根据输出矩阵的每一行预测下一个单词，如下图27所示。

![img](https://pic3.zhimg.com/80/v2-585526f8bfb9b4dfc691dfeb42562962_1440w.jpg)

图27：Softmax 根据输出矩阵的每一行预测下一个单词

如下图28所示为Transformer的整体结构。

![img](https://pic2.zhimg.com/80/v2-b9372cc3b3a810dba41e1a64d3b296d5_1440w.jpg)图28：Transformer的整体结构

#### 2.2 代码解读：

略


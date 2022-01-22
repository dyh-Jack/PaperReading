#### Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure

Centripetal SGD属于通道剪枝的方法

主要方法：不采用传统方法对重要性进行评估，而是对每一层的卷积核进行分组，并通过特殊的参数更新法则使得每组卷积核的参数逐渐趋同，最终从每个组取出一个卷积核，其余卷积核即可被剪枝。

引入$F^{(j)}$表示第j个卷积核的参数，$c_i，r_i$分别表示第i层原有卷积核数量和分组数量，$H(j)=H_2$代表第j个卷积核属于的组

![image-20220122230001536](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220122230001536.png)

参数更新法则如上。**其中，第一项是所在组内所有卷积核梯度的平均值，第二项为常规权值衰减，第三项为用于衡量组内卷积核相似度的数值项，其中超参数$\epsilon$称为向心强度**

其中引入X来衡量每组内卷积核的相似度

![image-20220122230456350](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220122230456350.png)

通过更新公式我们能够得到，随着迭代次数增加，X不断减小，相似度越来越高。

直观来说，更新公式中，**对于在同一簇的卷积核，由目标函数得到的增量（第一项）被平均，第二项是普通的权重衰减，初始值的差异逐渐消除（最后一项），所以卷积核逐渐向它们在超空间的中心靠近。**

一个例子可以帮助理解：

![image-20220122230856096](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220122230856096.png)

左边是标准的SGD，A是初始权重点，Q0和Q1代表误差梯度和权重衰减的方向，合成最终方向Q2。

右边将初始权重A和B分到了一个组，算出中心点M，在更新过程中因为增加了Q3方向（上文等式的第三项），更新方向变为Q4。结合公式也就是：

![image-20220122231125270](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220122231125270.png)

其中$\epsilon$可以理解成是一个控制A和B接近的强度或速度的超参数

![image-20220122231228318](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220122231228318.png)

![image-20220122231517642](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220122231517642.png)

![image-20220122231655636](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220122231655636.png)

**其中提到使用1/2偏导数和来替换，从而满足模型收敛要求，即是更新公式中第一项取平均值的原因。**

![image-20220122232403328](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220122232403328.png)

经过C-SGD训练后，由于每个簇中的卷积核都是相同的，选择哪一个没有区别。我们只需要在每个集群中选取第一个卷积核(即具有最小索引的卷积核)来形成每个层的剩余集合，即

![image-20220122232748339](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220122232748339.png)

![image-20220122232853323](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220122232853323.png)

![image-20220122233041408](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220122233041408.png)

如上图，对conv1中下方两个卷积核进行C-SGD，之后删除第四个卷积核，同时conv2层中每一个卷积核的第四层都加到第三层上，即完成对下一层的更新。

由于卷积核的线性和组合特性，这种变换不会损伤模型精度。（即原本一个卷积核在卷积时是第三层特征图和第三层卷积核卷积、第四层特征图和第四层卷积核卷积；现在变成第三层特征图和第三层卷积核卷积再加上第三层特征图和第四层卷积核卷积，一个卷积核最终得到的输出为一个新的特征图，和之前的表达能力相同）

基本原理如下：

![image-20220122233813996](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220122233813996.png)

$M^{(i,j)}$为第i层第j个通道，以此类推。

![image-20220122234526182](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220122234526182.png)

![image-20220122234537326](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220122234537326.png)
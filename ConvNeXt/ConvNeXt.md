#### ConvNeXt: A ConvNet for the 2020s

![image-20220124141158503](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220124141158503.png)

论文的整体思路是从原始的ResNet出发，通过借鉴Swin Transformer的设计来逐步地改进模型。论文共选择了两个不同大小的ResNet模型：**ResNet50和ResNet200**，其中ResNet50和Swin-T有类似的FLOPs（4G vs 4.5G），而ResNet200和Swin-B有类似的FLOPs（15G）。首先做的改进是**调整训练策略**，然后是模型设计方面的递进优化：**宏观设计->ResNeXt化->改用Inverted bottleneck->采用large kernel size->微观设计**。由于模型性能和FLOPs强相关，所以在优化过程中尽量保持FLOPs的稳定。 

#### 训练策略

舍弃原有的ResNet训练策略（**对于ResNet50，其训练策略比较简单**（torchvision版本）：batch size是32*8，epochs为90；优化器采用momentum=0.9的SGD，初始学习速率为0.1，然后每30个epoch学习速率衰减为原来的0.1；正则化只有L2，weight decay=1e-4；数据增强采用随机缩放裁剪（RandomResizedCrop）+水平翻转（RandomHorizontalFlip）），转而**采用DeiT的训练策略**：采用了比较多的数据增强如Mixup，Cutmix和RandAugment；训练的epochs增加至300；训练的optimizer采用AdamW，学习速率schedule采用cosine decay；采用smooth label和EMA等优化策略。这里直接将DeiT的训练策略（具体参数设置如下表）

![image-20220124155223106](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220124155223106.png)

#### 宏观设计

##### stage计算量占比

对比ResNet50，4个stage的blocks数量分别是（3，4，6，3），而Swin-T的设置为（2，2，6，2），4个stage的计算量比约为1:1:3:1。这里调整ResNet50各个stage的blocks数量以和Swin-T一致：从原来的（3，4，6，3）调整至（3，3，9，3）。调整之后模型性能从78.8%提升至79.4%

##### stem设计

第二个就是stem的区别。对于Swin-T模型，其stem是一个patch embedding layer，实际上就是一个stride=4的4x4 conv。而ResNet50的stem相对更复杂一些：首先是一个stride=2的7x7 conv，然后是一个stride=2的3x3 max pooling。两种stem最后均是得到1/4大小的特征，所以这里可以直接用Swin的stem来替换ResNet的stem，这个变动对模型效果影响较小：从79.4%提升至79.5%。

#### ResNeXt化

相比ResNet，ResNeXt通过采用group conv来提升性能，标准的conv其输入是所有的channels，而group conv会对channels进行分组来减少计算量，这样节省下来的计算量用来增加网络的width即特征channels。对于group conv，其最极端的情况就是每个channel一个group，这样就变成了depthwise conv（简称dw conv），dw conv首先在MobileNet中应用，后来也被其它CNN模型广泛采用。对于dw conv，其和local attention有很多的相似的地方，local attention其实就是对window里的各个token的特征做加权和，而且操作是per-channel的；而dw conv是对kernel size范围的token的特征求加权和，也是分channel的。这里的最大区别就是：self-attention的权重是动态计算的（data dependent），而dw conv的权重就是学习的kernel。

这里将ResNet50中的3x3 conv替换成3x3 dw conv，为了弥补FLOPs的减少，同时将ResNet50的base width从原来的64增加至96（和Swin-T一致，这里的base width是指stem后的特征大小），此时模型的FLOPs有所增加（5.3G），模型性能提升至80.5%。

![preview](https://pic4.zhimg.com/v2-ad0191830458a8368bd8fff386708cbf_r.jpg)

#### Inverted Bottleneck

如果把self-attention看成一个dw conv的话（这里忽略self-attention的linear projection操作），那么一个transformer block可以近似看成一个inverted bottleneck，因为MLP等效于两个1x1 conv，并且MLP中间隐含层特征是输入特征大小的4倍（expansion ratio=4）。inverted bottleneck最早在MobileNetV2中提出，随后的EfficientNet也采用了这样的结构。ResNet50采用的是正常的residual bottleneck，这里将其改成inverted bottleneck，即从图（a）变成图（b），虽然dw conv的计算量增加了，但是对于包含下采样的residual block中，用于shortcut的1x1 conv计算量却大大降低，最终模型的FLOPs减少为4.6G。这个变动对ResNet50的影响较小（80.5%->80.6%)。

![image-20220124160121976](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220124160121976.png)

Large Kernel Size

自从VGG之后，主流的CNN往往采用较小的kernel size，如3x3和5x5，其中3x3 conv在GPU上有高效的实现。然而Swin-T采用的window size为7x7，这比3x3 conv对应的windwo size要大得多，所以这里考虑采用更大的kernel size。 在这之前，首先**将dw conv移到inverted bottleneck block的最开始**，如上图（c）所示。对于transformer block，其实self-attention也是在开始，同时由于采用inverted bottleneck，将dw conv移动到最前面可以减少计算量（4.1G），后续采用较大的kernel size后模型的FLOPs变动更少。由于模型FLOPs的降低，模型性能也出现一定的下降：80.6%->79.9%。 然后调整dw conv的kernel size，这里共实验了5种kernel size：3x3，5x5，7x7，9x9和11x11。实验发现kernel size增加，模型性能有提升，但是在7x7之后采用更大的kernel size性能达到饱和。所以最终选择7x7，这样也和Swin-T的window size一致，由于前面的dw conv位置变动，采用7x7的kernel size基本没带来FLOPs的增加。采用7x7 dw conv之后，模型的性能又回到80.6%。

#### 微观设计

经过前面的改动，模型的性能已经提升到80%以上，此时改动后的ResNet50也和Swin-T在整体结构上很类似了，下面我们开始关注一些微观设计上的差异，或者说是layer级别的不同。 首先是激活函数，**CNN模型一般采用ReLU，而transformer模型常常采用GELU**，两者的区别如下图所示。这里把激活函数都从ReLU改成GELU，模型效果没有变化（80.6%）。

![preview](https://pic1.zhimg.com/v2-26b5573ce3947fe1299b9d0980f5f4a4_r.jpg)

另外的一个差异是transformer模型只在MLP中间采用了非线性激活，而CNN模型常常每个conv之后就会用跟一个非线性激活函数。如下图示，这里**只保留中间1x1 conv之后的GELU**，就和Swin-T基本保持一致了，这个变动使模型性能从80.6%提升至81.3%。

![image-20220124160453298](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220124160453298.png)

对于norm层，也存在和激活函数一样的问题，transformer中只在self-attention和MLP的开始采用了LayerNorm，而ResNet每个conv之后采用BatchNorm，比transformer多一个norm层。这里去掉其它的BatchNorm，**只保留中间1x1 conv前的BatchNorm**，此时模型性能有0.1%的提升。实际上要和transformer保持一致，应该在block最开始增加一个BatchNorm，但是这个并没有提升性能，所以最终只留下了一个norm层。另外，transformer的norm层采用LayerNorm，而CNN常采用BatchNorm，一般情况下BatchNorm要比LayerNorm效果要好，但是BatchNorm受batch size的影响较大。**这里将BatchNorm替换成LayerNorm后，模型性能只有微弱的下降（80.5%）**。 最后一个差异是下采样，ResNet中的下采样一般放在每个stage的最开始的block中，采用stride=2的3x3 conv；但是Swin-T**采用分离的下采样**，即下采样是放在两个stage之间，通过一个stride=2的2x2 conv（论文中叫patch merging layer）。但是实验发现，如果直接改用Swin-T的下采样，会出现训练发散问题，解决的办法是在添加几个norm层：**在stem之后，每个下采样层之前以及global avg pooling之后都增加一个LayerNom**（Swin-T也是这样做的）。最终模型的性能提升至82.0%，超过Swin-T（81.3%）。

经过6个方面的修改或者优化，最终得到的模型称为**ConvNeXt**，其模型结构如下所示，可以看到，ConvNeXt-T和Swin-T整体结构基本一致，而且模型FLOPs和参数量也基本一样，**唯一的差异就是dw conv和W-MSA**（MSA的前面和后面都包含linear projection，等价1x1 conv），由于dw conv和W-MSA的计算量存在差异，所以ConvNeXt-T比Swin-T包含更多的blocks。另外MSA是permutation-invariance的，所以W-MSA采用了相对位置编码；而且Swin-T需要采用shifted window来实现各个windows间的信息交互；而相比之下，ConvNeXt-T不需要这样的特殊处理。

![image-20220124160745510](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220124160745510.png)

#### 小结

从创新上看，ConvNeXt并没有太多的新意，看起来就是把vision transformer一些优化搬到了CNN上，而且之前也有很多类似的工作，但我认为ConvNeXt是一个很好的工作，因为做了比较全面的实现，而且ConvNeXt在工业部署上更有价值，因为swin transformer的实现太过tricky。从CNN到vision transformer再到CNN，还包括中间对MLP的探索，或许我们越来越能得出结论：**架构没有优劣之分，在同样的FLOPs下，不同的模型的性能是接近的**。但在实际任务上，受限于各种条件，我们可能看到不同模型上的性能差异，这就需要具体问题具体分析了。
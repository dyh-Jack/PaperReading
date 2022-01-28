#### Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

Swin Transformer总体架构：

![image-20220126210831793](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220126210831793.png)

#### 基本名词解释

- pactch：图像4x4像素区域称为一个patch，分类任务输入图像像素分辨率是224x224，patch_size = 4，所以patches__resolution = 56*56
- 窗口大小由patches定义的，不是像素定义的，论文及程序中window_size = 7，说明一个window有7*7=49个patches

#### 模型架构

Swin Transformer提出了一种具有滑动窗口的层级设计Transformer架构。

其中**滑窗操作**包括**不重叠的 local window，和重叠的 cross-window**。将注意力计算限制在一个窗口（window size固定）中，**一方面能引入 CNN 卷积操作的局部性，另一方面能大幅度节省计算量**，它只和窗口数量成线性关系。

通过**下采样**的层级设计，能够逐渐增大感受野，从而使得注意力机制也能够注意到**全局**的特征。

![image-20220126213327756](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220126213327756.png)

整个模型采取层次化的设计，一共包含 4 个 Stage，除第一个 stage 外，每个 stage 都会先通过 **Patch Merging** 层缩小输入特征图的分辨率，进行**下采样操作**，像 CNN 一样逐层扩大感受野，以便获取到全局的信息。

Swin Transformer的总体架构：

- 在输入开始的时候，做了一个`Patch Partition`，即ViT中`Patch Embedding`操作，通过 **Patch_size** 为4的卷积层将图片切成一个个 **Patch** ，并嵌入到`Embedding`，将 **embedding_size** 转变为48（可以将 CV 中图片的**通道数**理解为NLP中token的**词嵌入长度**）。
- 随后在第一个Stage中，通过`Linear Embedding`调整通道数为C。
- 在每个 Stage 里（除第一个 Stage ），均由`Patch Merging`和多个`Swin Transformer Block`组成。
- 其中`Patch Merging`模块主要在每个 Stage 一开始降低图片分辨率，进行下采样的操作。
- 而`Swin Transformer Block`具体结构如右图所示，主要是`LayerNorm`，`Window Attention` ，`Shifted Window Attention`和`MLP`组成 。

在微软亚洲研究院提供的代码中，是将`Patch Merging`作为每个 Stage 最后结束的操作，输入先进行`Swin Transformer Block`操作，再下采样。而**最后一个 Stage 不需要进行下采样操作**，之间通过后续的全连接层与 **target label** 计算损失。

#### Patch Embedding

在输入进 Block 前，我们需要将图片切成一个个 patch，然后嵌入向量。**（如若输入为224x224，patch_size为4x4，那么每一个patch应为56x56）**

具体做法是对原始图片裁成一个个 56x56的窗口大小，然后进行嵌入。

这里可以通过二维卷积层，**将 stride，kernel_size 设置为 patch_size 大小**。设定输出通道来确定嵌入向量的大小。最后将 H,W 维度展开，并移动到第一维度。

```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size) # -> (img_size, img_size)
        patch_size = to_2tuple(patch_size) # -> (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # 假设采取默认参数，论文中embedding_size是48，但是代码中为96.我们以代码为准
        x = self.proj(x) # 出来的是(N, 96, 224/4, 224/4) 
        x = torch.flatten(x, 2) # 把HW维展开，(N, 96, 56*56)
        x = torch.transpose(x, 1, 2)  # 把通道维放到最后 (N, 56*56, 96)
        if self.norm is not None:
            x = self.norm(x)
        return x
```

#### Patch Merging

该模块的作用是在每个 Stage 开始前做降采样，用于缩小分辨率，调整通道数进而形成层次化的设计，同时也能节省一定运算量。

其中，操作示意图如下：

![preview](https://pic2.zhimg.com/v2-e62428c0dec724aa8e30c27c49c29131_r.jpg)

```python
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
		
        # 0::2 偶数
        # 1::2 奇数
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
```

整体流程如下：

![preview](https://pic4.zhimg.com/v2-036755666896f4e60ff9ad8d129f2ca3_r.jpg)

#### Window Partition/Reverse

`window partition`函数是用于对张量划分窗口，指定窗口大小。将原本的张量从 `N H W C`, 划分成 `num_windows*B, window_size, window_size, C`，其中 `num_windows = H*W / window_size*window_size`，即窗口的个数。而`window reverse`函数则是对应的逆过程。这两个函数会在后面的`Window Attention`用到。

![preview](https://pic1.zhimg.com/v2-f44fcd9873740f95dc3363349f4c8604_r.jpg)

```python
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
```

#### Window Attention

传统的 Transformer 都是**基于全局来计算注意力的**，因此计算复杂度十分高。而 Swin Transformer 则将**注意力的计算限制在每个窗口内**，进而减少了计算量。

公式如下：

![image-20220126234015657](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220126234015657.png)

主要区别是在原始计算 Attention 的公式中的 Q,K 时**加入了相对位置编码**。

```python
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads # nH
        head_dim = dim // num_heads # 每个注意力头对应的通道数
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 设置一个形状为（2*(Wh-1) * 2*(Ww-1), nH）的可学习变量，用于后续的位置编码

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        # 相关位置编码...
```

##### 相对位置编码

```python
Q,K,V.shape=[numWindwos*B, num_heads, window_size*window_size, head_dim]
```

- window_size*window_size 即 NLP 中`token`的个数
- ![[公式]](https://www.zhihu.com/equation?tex=head%5C_dim%3D%5Cfrac%7BEmbedding%5C_dim%7D%7Bnum%5C_heads%7D) 即 NLP 中`token`的词嵌入向量的维度

![[公式]](https://www.zhihu.com/equation?tex=%7BQK%7D%5ET)计算出来的`Attention`张量的形状为`[numWindows*B, num_heads, Q_tokens, K_tokens]`

- 其中Q_tokens=K_tokens=window_size*window_size

![preview](https://pic1.zhimg.com/v2-f489a26ca0765c5e11ace0dc1613e4f0_r.jpg)

因此：![[公式]](https://www.zhihu.com/equation?tex=%7BQK%7D%5ET%3D%5Cleft%5B%5Cbegin%7Barray%7D%7Bcccc%7Da_%7B11%7D+%26+a_%7B12%7D+%26+a_%7B13%7D+%26+a_%7B14%7D+%5C%5C+a_%7B21%7D+%26+a_%7B22%7D+%26+a_%7B23%7D+%26+a_%7B24%7D+%5C%5C+a_%7B31%7D+%26+a_%7B32%7D+%26+a_%7B33%7D+%26+a_%7B34%7D+%5C%5C+a_%7B41%7D+%26+a_%7B42%7D+%26+a_%7B43%7D+%26+a_%7B44%7D%5Cend%7Barray%7D%5Cright%5D)

- **第 ![[公式]](https://www.zhihu.com/equation?tex=i) 行表示第 ![[公式]](https://www.zhihu.com/equation?tex=i) 个 token 的`query`对所有token的`key`的attention。**
- 对于 Attention 张量来说，**以不同元素为原点，其他元素的坐标也是不同的**，

![preview](https://pic3.zhimg.com/v2-bc5414042259123ddf67a5ab77322a3a_r.jpg)

所以![[公式]](https://www.zhihu.com/equation?tex=%7BQK%7D%5ET%E7%9A%84%E7%9B%B8%E5%AF%B9%E4%BD%8D%E7%BD%AE%E7%B4%A2%E5%BC%95%3D%5Cleft%5B%5Cbegin%7Barray%7D%7Bcccc%7D%280%2C0%29+%26+%280%2C-1%29+%26+%28-1%2C0%29+%26+%28-1%2C-1%29+%5C%5C+%280%2C1%29+%26+%280%2C0%29+%26+%28-1%2C1%29+%26+%28-1%2C0%29+%5C%5C+%281%2C0%29+%26+%281%2C-1%29+%26+%280%2C0%29+%26+%280%2C-1%29+%5C%5C+%281%2C1%29+%26+%281%2C0%29+%26+%280%2C1%29+%26+%280%2C0%29%5Cend%7Barray%7D%5Cright%5D)

由于最终我们希望使用一维的位置坐标 `x+y` 代替二维的位置坐标`(x,y)`，为了避免 (1,2) (2,1) 两个坐标转为一维时均为3，我们之后对相对位置索引进行了一些**线性变换**，使得能通过**一维**的位置坐标**唯一映射**到一个**二维**的位置坐标，详细可以通过代码部分进行理解。

代码分析：

首先我们利用`torch.arange`和`torch.meshgrid`函数生成对应的坐标，这里我们以`windowsize=2`为例子

```python
coords_h = torch.arange(self.window_size[0])
coords_w = torch.arange(self.window_size[1])
coords = torch.meshgrid([coords_h, coords_w]) # -> 2*(wh, ww)
"""
  (tensor([[0, 0],
           [1, 1]]), 
   tensor([[0, 1],
           [0, 1]]))
"""
```

然后堆叠起来，展开为一个二维向量

```python
coords = torch.stack(coords)  # 2, Wh, Ww
coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
"""
tensor([[0, 0, 1, 1],
        [0, 1, 0, 1]])
"""
```

利用广播机制，分别在第一维，第二维，插入一个维度，进行广播相减，得到 `2, wh*ww, wh*ww`的张量

```python
relative_coords_first = coords_flatten[:, :, None]  # 2, wh*ww, 1
relative_coords_second = coords_flatten[:, None, :] # 2, 1, wh*ww
relative_coords = relative_coords_first - relative_coords_second # 最终得到 2, wh*ww, wh*ww 形状的张量
```

![preview](https://pic2.zhimg.com/v2-1266748e072dff0effeabebbbd159f35_r.jpg)

因为采取的是相减，所以得到的索引是从负数开始的，**我们加上偏移量，让其从 0 开始**。

```python
relative_coords = relative_coords.permute(1, 2, 0).contiguous() # Wh*Ww, Wh*Ww, 2
relative_coords[:, :, 0] += self.window_size[0] - 1
relative_coords[:, :, 1] += self.window_size[1] - 1
```

后续我们需要将其展开成一维偏移量。而对于 (1，2）和（2，1）这两个坐标。在二维上是不同的，**但是通过将 x,y 坐标相加转换为一维偏移的时候，他的偏移量是相等的**。

![preview](https://pic4.zhimg.com/v2-29a58ceb2b37d998a6c9ead526011b33_r.jpg)

所以最后我们对其中做了个乘法操作，以进行区分

```python
relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
```

![preview](https://pic2.zhimg.com/v2-404beb4c31ea9570e7d915f3299ecef1_r.jpg)

然后再最后一维上进行求和，展开成一个一维坐标，并注册为一个不参与网络学习的变量

```python
relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
self.register_buffer("relative_position_index", relative_position_index)
```

之前计算的是相对位置索引，并不是相对位置偏置参数。真正使用到的可训练参数![[公式]](https://www.zhihu.com/equation?tex=%5Chat+B)是保存在`relative position bias table`表里的，这个表的长度是等于 **(2M−1) × (2M−1)** **<u>(在二维位置坐标中线性变化乘以2M-1导致，可列式计算)</u>**的。那么上述公式中的相对位置偏执参数B是根据上面的相对位置索引表根据查`relative position bias table`表得到的。

![preview](https://pic4.zhimg.com/v2-1906973885e00f296df1a71502f3c76b_r.jpg)

前向传播代码为：

```python
def forward(self, x, mask=None):
    """
    Args:
        x: input features with shape of (num_windows*B, N, C)
        mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
    """
    B_, N, C = x.shape

    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

    q = q * self.scale
    attn = (q @ k.transpose(-2, -1))

    relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    attn = attn + relative_position_bias.unsqueeze(0) # (1, num_heads, windowsize, windowsize)

    if mask is not None: # 下文会分析到
        ...
    else:
        attn = self.softmax(attn)

    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x
```

- 首先输入张量形状为 `[numWindows*B, window_size * window_size, C]`
- 然后经过`self.qkv`这个全连接层后，进行 reshape，调整轴的顺序，得到形状为`[3, numWindows*B, num_heads, window_size*window_size, c//num_heads]`，并分配给`q,k,v`。
- 根据公式，我们对`q`乘以一个`scale`缩放系数，然后与`k`（为了满足矩阵乘要求，需要将最后两个维度调换）进行相乘。得到形状为`[numWindows*B, num_heads, window_size*window_size, window_size*window_size]`的`attn`张量
- 之前我们针对位置编码设置了个形状为`(2*window_size-1*2*window_size-1, numHeads)`的可学习变量。我们用计算得到的相对编码位置索引`self.relative_position_index.vew(-1)`选取，得到形状为`(window_size*window_size,window_size*window_size, numHeads)`的编码，再permute(2,0,1)后加到`attn`张量上
- 暂不考虑 mask 的情况，剩下就是跟 transformer 一样的 softmax，dropout，与`V`矩阵乘，再经过一层全连接层和 dropout

#### Shifted Window Attention

前面的 Window Attention 是在每个窗口下计算注意力的，为了更好的和其他 window 进行信息交互，Swin Transformer 还引入了 shifted window 操作。

![image-20220127001848014](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220127001848014.png)

左边是没有重叠的 Window Attention，而右边则是将窗口进行移位的 Shift Window Attention。可以看到移位后的窗口包含了原本相邻窗口的元素。但这也引入了一个新问题，即 **window 的个数增加了**，由原本四个窗口变成了 9 个窗口。

在实际代码里，我们是**通过对特征图移位，并给 Attention 设置 mask 来间接实现的**。能在**保持原有的 window 个数下**，最后的计算结果等价。

通过下图的方式可以在保证不重叠窗口间有联系的基础上不增加窗口的个数，新的窗口可能会由之前不相关的自窗口构成，为了保证shifted window self-attention计算的正确性，只能计算相同子窗口的self-attention，不同子窗口的self-attention结果要归0，否则就违背了shifted window self-attention 的原则。

![image-20220127160459567](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220127160459567.png)

##### 特征图位移

Shifted Window方法是在连续的两个Transformer Block之间实现的。

- 第一个模块使用一个标准的window partition策略，从feature map的左上角出发，例如一个8*8的feature map会被平分为2*2个window，每个window的大小为![[公式]](https://www.zhihu.com/equation?tex=M%3D4)。
- 紧接着的第二个模块则使用了移动窗口的策略，window会从feature map的![[公式]](https://www.zhihu.com/equation?tex=%5Cleft%28%5Cleft%5Clfloor%5Cfrac%7BM%7D%7B2%7D%5Cright%5Crfloor%2C%5Cleft%5Clfloor%5Cfrac%7BM%7D%7B2%7D%5Cright%5Crfloor%5Cright%29)位置处开始，然后再进行window partition操作。

这样一来，不同window之间在两个连续的模块之间便有机会进行交互。

然而，Shifted Window Partition存在一个问题，由于没有与边界对齐，其会产生更多的Windows，从![[公式]](https://www.zhihu.com/equation?tex=%5Cleft%5Clceil%5Cfrac%7Bh%7D%7BM%7D%5Cright%5Crceil+%5Ctimes%5Cleft%5Clceil%5Cfrac%7Bw%7D%7BM%7D%5Cright%5Crceil)个Windows上升至![[公式]](https://www.zhihu.com/equation?tex=%5Cleft%5Clceil%5Cfrac%7Bh%7D%7BM%7D%2B1%5Cright%5Crceil+%5Ctimes%5Cleft%5Clceil%5Cfrac%7Bw%7D%7BM%7D%2B1%5Cright%5Crceil)，并且其中很多windows的大小也不足![[公式]](https://www.zhihu.com/equation?tex=M%2AM)

这其中，可以有一种比较naive的方法：

![preview](https://pic3.zhimg.com/v2-c7d037d20ef14a5c0098572e38cc2bd2_r.jpg)

可以看出这种解决方法的缺点在于额外计算了很多padding的部分，浪费了大量计算。

而论文提出了一种新的方法：

![preview](https://pic2.zhimg.com/v2-c6cf952e2981cdd77661ac09e4de1fe9_r.jpg)

##### Attention Mask

这是 Swin Transformer 的精华，通过设置合理的 mask，让`Shifted Window Attention`在与`Window Attention`相同的窗口个数下，达到等价的计算结果。（**<u>其实本质上还是计算的shifted window的attention，只是在mask之后可以在节省大量计算量的同时满足对于新的窗口attention的计算</u>**）

首先我们对 Shift Window 后的每个窗口都给上 index，并且做一个`roll`操作（window_size=2, shift_size=1）

我们希望在计算 Attention 的时候，**让具有相同 index QK 进行计算，而忽略不同 index QK 计算结果**。

在(5,3)\(7,1)\(8,6,2,0)组成的新窗口中，只有相同编码的部分才能计算self-attention，不同编码位置间计算的self-attention需要归0，根据self-attention公式，最后需要进行Softmax操作，不同编码位置间计算的self-attention结果通过mask加上-100，在Softmax计算过程中，Softmax(-100)无线趋近于0，达到归0的效果。

![preview](https://pic4.zhimg.com/v2-a058ef6769f85a3c8394ef79b37c7447_r.jpg)

按照上图的设计，我们会得到这样的mask矩阵：

```text
tensor([[[[[   0.,    0.,    0.,    0.],
           [   0.,    0.,    0.,    0.],
           [   0.,    0.,    0.,    0.],
           [   0.,    0.,    0.,    0.]]],


         [[[   0., -100.,    0., -100.],
           [-100.,    0., -100.,    0.],
           [   0., -100.,    0., -100.],
           [-100.,    0., -100.,    0.]]],


         [[[   0.,    0., -100., -100.],
           [   0.,    0., -100., -100.],
           [-100., -100.,    0.,    0.],
           [-100., -100.,    0.,    0.]]],


         [[[   0., -100., -100., -100.],
           [-100.,    0., -100., -100.],
           [-100., -100.,    0., -100.],
           [-100., -100., -100.,    0.]]]]])
```

相关代码如下：

```python
if self.shift_size > 0:
    # calculate attention mask for SW-MSA
    H, W = self.input_resolution
    img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
    h_slices = (slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None))
    w_slices = (slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1 #对新的区块进行编号

    mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
else:
 attn_mask = None
```

这其中，slice进行切片，其中，如果按照上图设计，window_size=2（每个窗口中4个patch），共划分为4个窗口，shift_size = 1，即将原有的窗口整体向左上角移动一个单位，即可得到移动之后的新的窗口划分；在代码中直接进行切片得到对应的新的区块。

其中：

![[公式]](https://www.zhihu.com/equation?tex=shift%5C_size%3D%5Cleft%5Clfloor%5Cfrac%7BM%7D%7B2%7D%5Cright%5Crfloor)

![preview](https://pic4.zhimg.com/v2-dc2fe96c5c67510aeabbcff3489c9757_r.jpg)

![preview](https://pic1.zhimg.com/v2-61469886ee2ef8995996a5a0acd69ab8_r.jpg)

![img](https://pic1.zhimg.com/80/v2-9cb8b56e82d02370c8b243a54a5efc00_1440w.jpg)

![image-20220127193546715](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220127193546715.png)

![img](https://pic1.zhimg.com/80/v2-d4bd615abc8547385415512c4d4e0470_1440w.jpg)

![img](https://pic3.zhimg.com/80/v2-b483bbdff8181f3bef9c5445d86b2d36_1440w.jpg)

**相当于在经过shift_window之后，在计算attention的时候还是始终是相同编号的进行计算，编号不同的部分都要被掩码掩盖。**

在之前的 window attention 模块的前向代码里，包含这么一段

```python
if mask is not None:
    nW = mask.shape[0] # 一张图被分为多少个windows
    attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0) # torch.Size([128, 4, 12, 49, 49]) torch.Size([1, 4, 1, 49, 49])
    attn = attn.view(-1, self.num_heads, N, N)
    attn = self.softmax(attn)
else:
    attn = self.softmax(attn)
```

将 mask 加到 attention 的计算结果，并进行 softmax。mask 的值设置为 - 100，softmax 后就会忽略掉对应的值。

#### W-MSA和MSA的复杂度对比

##### MSA模块的计算量

![image-20220127163153570](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220127163153570.png)

- **h**：feature map的高度
- **w**：feature map的宽度
- **C**：feature map的通道数（也可以称为embedding size的大小）
- **M**：window_size的大小

首先对于`feature map`中每一个`token`（一共有 ![[公式]](https://www.zhihu.com/equation?tex=hw) 个token，通道数为C），记作![[公式]](https://www.zhihu.com/equation?tex=X%5E%7Bh+w+%5Ctimes+C%7D)，需要通过三次线性变换 ![[公式]](https://www.zhihu.com/equation?tex=W_q%2CW_k%2CW_v) ，产生对应的`q,k,v`向量，记作 ![[公式]](https://www.zhihu.com/equation?tex=Q%5E%7Bh+w+%5Ctimes+C%7D%2CK%5E%7Bh+w+%5Ctimes+C%7D%2CV%5E%7Bh+w+%5Ctimes+C%7D) （通道数为C)。

![[公式]](https://www.zhihu.com/equation?tex=X%5E%7Bh+w+%5Ctimes+C%7D+%5Ccdot+W_%7Bq%7D%5E%7BC+%5Ctimes+C%7D%3DQ%5E%7Bh+w+%5Ctimes+C%7D+%5C%5C+X%5E%7Bh+w+%5Ctimes+C%7D+%5Ccdot+W_%7Bk%7D%5E%7BC+%5Ctimes+C%7D%3DK%5E%7Bh+w+%5Ctimes+C%7D+%5C%5C+X%5E%7Bh+w+%5Ctimes+C%7D+%5Ccdot+W_%7Bv%7D%5E%7BC+%5Ctimes+C%7D%3DV%5E%7Bh+w+%5Ctimes+C%7D+%5C%5C+%5C%5C)

根据矩阵运算的计算量公式可以得到运算量为 ![[公式]](https://www.zhihu.com/equation?tex=3hwC+%5Ctimes+C) ，即为 ![[公式]](https://www.zhihu.com/equation?tex=3hwC%5E2) 。

![[公式]](https://www.zhihu.com/equation?tex=Q%5E%7Bh+w+%5Ctimes+C%7D+%5Ccdot+K%5ET%3DA%5E%7Bh+w+%5Ctimes+hw%7D+%5C%5C+%5CLambda%5E%7Bh+w+%5Ctimes+h+w%7D%3DSoftmax%28%5Cfrac%7BA%5E%7Bh+w+%5Ctimes+hw%7D%7D%7B%5Csqrt%28d%29%7D%2BB%29+%5C%5C+%5CLambda%5E%7Bh+w+%5Ctimes+h+w%7D+%5Ccdot+V%5E%7Bh+w+%5Ctimes+C%7D%3DY%5E%7Bh+w+%5Ctimes+C%7D+%5C%5C)

忽略除以![[公式]](https://www.zhihu.com/equation?tex=%5Csqrt+d) 以及softmax的计算量，根据根据矩阵运算的计算量公式可得 ![[公式]](https://www.zhihu.com/equation?tex=hwC+%5Ctimes+hw+%2B+hw%5E2+%5Ctimes+C) ，即为 ![[公式]](https://www.zhihu.com/equation?tex=2%28hw%5E2%29C) 。

![[公式]](https://www.zhihu.com/equation?tex=Y%5E%7Bh+w+%5Ctimes+C%7D+%5Ccdot+W_O%5E%7BC+%5Ctimes+C%7D%3DO%5E%7Bh+w+%5Ctimes+C%7D+%5C%5C)

最终再通过一个Linear层输出，计算量为 ![[公式]](https://www.zhihu.com/equation?tex=hwC%5E2) 。因此整体的计算量为 ![[公式]](https://www.zhihu.com/equation?tex=4+h+w+C%5E%7B2%7D%2B2%28h+w%29%5E%7B2%7D+C) 。

##### **W-MSA模块的计算量**

对于W-MSA模块，首先会将`feature map`根据`window_size`分成 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7Bhw%7D%7BM%5E2%7D) 的窗口，每个窗口的宽高均为![[公式]](https://www.zhihu.com/equation?tex=M)，然后在每个窗口进行MSA的运算。因此，可以利用上面MSA的计算量公式，将 ![[公式]](https://www.zhihu.com/equation?tex=h%3DM%EF%BC%8Cw%3DM) 带入，可以得到一个窗口的计算量为 ![[公式]](https://www.zhihu.com/equation?tex=4+M%5E2+C%5E%7B2%7D%2B2M%5E%7B4%7D+C) 。

又因为有 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7Bhw%7D%7BM%5E2%7D) 个窗口，则：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7Bhw%7D%7BM%5E2%7D+%5Ctimes%5Cleft%284M%5E2+C%5E2%2B2M%5E%7B4%7D+C%5Cright%29%3D4+h+w+C%5E%7B2%7D%2B2+M%5E%7B2%7D+h+w+C+%5C%5C)

假设`feature map`的![[公式]](https://www.zhihu.com/equation?tex=h%3Dw%3D112%EF%BC%8CM%3D7%EF%BC%8CC%3D128)，采用W-MSA模块会比MSA模块节省约40124743680 FLOPs：

![[公式]](https://www.zhihu.com/equation?tex=2%28h+w%29%5E%7B2%7D+C-2+M%5E%7B2%7D+h+w+C%3D2+%5Ctimes+112%5E%7B4%7D+%5Ctimes+128-2+%5Ctimes+7%5E%7B2%7D+%5Ctimes+112%5E%7B2%7D+%5Ctimes+128%3D40124743680+%5C%5C)

#### 整体流程

SwinT：

![preview](https://pic3.zhimg.com/v2-cb6064a72fbbe684a95eb849a529f342_r.jpg)

SwinT网络：

![preview](https://pic3.zhimg.com/v2-d3c0b8ef64543b4d7c67a98f023b21f2_r.jpg)
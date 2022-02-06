#### An Image is Worth 16x16 Words:Transformers for Image Recognition at Scale

#### ViT原理分析：

这个工作本着尽可能少修改的原则，将原版的Transformer开箱即用地迁移到分类任务上面。并且作者认为没有必要总是依赖于CNN，只用Transformer也能够在分类任务中表现很好，尤其是在使用大规模训练集的时候。同时，在大规模数据集上预训练好的模型，在迁移到中等数据集或小数据集的分类任务上以后，也能取得比CNN更优的性能。**下面看具体的方法：**

**图片预处理：分块和降维**

这个工作首先把![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bx%7D%5Cin+H%5Ctimes+W%5Ctimes+C)的图像，变成一个 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bx%7D_p%5Cin+N%5Ctimes+%28P%5E2%5Ccdot+C%29) 的sequence of flattened 2D patches。它可以看做是一系列的展平的2D块的序列，这个序列中一共有 ![[公式]](https://www.zhihu.com/equation?tex=N%3DHW%2FP%5E2) 个展平的2D块，每个块的维度是 ![[公式]](https://www.zhihu.com/equation?tex=%28P%5E2%5Ccdot+C%29) 。其中 ![[公式]](https://www.zhihu.com/equation?tex=P) 是块大小， ![[公式]](https://www.zhihu.com/equation?tex=C) 是channel数。

**注意作者做这步变化的意图：**根据我们之前的讲解，Transformer希望输入一个二维的矩阵 ![[公式]](https://www.zhihu.com/equation?tex=%28N%2CD%29) ，其中 ![[公式]](https://www.zhihu.com/equation?tex=N) 是sequence的长度， ![[公式]](https://www.zhihu.com/equation?tex=D) 是sequence的每个向量的维度，常用256。

所以这里也要设法把 ![[公式]](https://www.zhihu.com/equation?tex=H%5Ctimes+W%5Ctimes+C) 的三维图片转化成 ![[公式]](https://www.zhihu.com/equation?tex=%28N%2CD%29) 的二维输入。

所以有： ![[公式]](https://www.zhihu.com/equation?tex=H%5Ctimes+W%5Ctimes+C%5Crightarrow+N%5Ctimes+%28P%5E2%5Ccdot+C%29%2C%5Ctext%7Bwhere%7D%5C%3BN%3DHW%2FP%5E2) 。

**其中，![[公式]](https://www.zhihu.com/equation?tex=N) 是Transformer输入的sequence的长度。**

代码是：

```python3
x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
```

具体是采用了einops库实现，具体可以参考这篇博客。[PyTorch 70.einops：优雅地操作张量维度 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/342675997)

现在得到的向量维度是： ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bx%7D_p%5Cin+N%5Ctimes+%28P%5E2%5Ccdot+C%29) ，**要转化成 ![[公式]](https://www.zhihu.com/equation?tex=%28N%2CD%29) 的二维输入，我们还需要做一步叫做Patch Embedding的步骤。**

#### **Patch Embedding**

方法是对每个向量都做**一个线性变换（即全连接层）**，压缩后的维度为 ![[公式]](https://www.zhihu.com/equation?tex=D) ，这里我们称其为 Patch Embedding。

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bz%7D_0+%3D+%5B+%5Ccolor%7Bdarkgreen%7D%7B%5Cmathbf%7Bx%7D_%5Ctext%7Bclass%7D%7D%3B+%5C%2C+%5Ccolor%7Bcrimson+%7D%7B%5Cmathbf%7Bx%7D%5E1_p+%5Cmathbf%7BE%7D%3B+%5C%2C+%5Cmathbf%7Bx%7D%5E2_p+%5Cmathbf%7BE%7D%3B+%5Ccdots%3B+%5C%2C+%5Cmathbf%7Bx%7D%5E%7BN%7D_p+%5Cmathbf%7BE%7D+%7D%5D+%2B+%5Cmathbf%7BE%7D_%7Bpos%7D+%5Ctag%7B5.1%7D)

这个全连接层就是上式(5.1)中的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ccolor%7Bcrimson%7D%7B%5Cmathbf%7BE%7D%7D) ，它的输入维度大小是 ![[公式]](https://www.zhihu.com/equation?tex=%28P%5E2%5Ccdot+C%29) ，输出维度大小是 ![[公式]](https://www.zhihu.com/equation?tex=D)。

```python3
# 将3072变成dim，假设是1024
self.patch_to_embedding = nn.Linear(patch_dim, dim)
x = self.patch_to_embedding(x)
```

注意这里的绿色字体 ![[公式]](https://www.zhihu.com/equation?tex=%5Ccolor%7Bdarkgreen%7D%7B%5Cmathbf%7Bx%7D_%5Ctext%7Bclass%7D%7D) ，假设切成9个块，但是最终到Transfomer输入是10个向量，这是人为增加的一个向量。

**为什么要追加这个向量？**

如果没有这个向量，假设 ![[公式]](https://www.zhihu.com/equation?tex=N%3D9) 个向量输入Transformer Encoder，输出9个编码向量，然后呢？对于分类任务而言，我应该取哪个输出向量进行后续分类呢？

不知道。干脆就再来一个向量 ![[公式]](https://www.zhihu.com/equation?tex=%5Ccolor%7Bdarkgreen%7D%7B%5Cmathbf%7Bx%7D_%5Ctext%7Bclass%7D%7D%28%5Ctext%7Bvector%2Cdim%7D%3DD%29) ，这个向量是**可学习的嵌入向量**，它和那9个向量一并输入Transfomer Encoder，输出1+9个编码向量。然后就用第0个编码向量，即 ![[公式]](https://www.zhihu.com/equation?tex=%5Ccolor%7Bdarkgreen%7D%7B%5Cmathbf%7Bx%7D_%5Ctext%7Bclass%7D%7D) 的输出进行分类预测即可。

这么做的原因可以理解为：ViT其实只用到了Transformer的Encoder，而并没有用到Decoder，而 ![[公式]](https://www.zhihu.com/equation?tex=%5Ccolor%7Bdarkgreen%7D%7B%5Cmathbf%7Bx%7D_%5Ctext%7Bclass%7D%7D) 的作用有点类似于解码器中的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BQuery%7D) 的作用，相对应的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BKey%2CValue%7D) 就是其他9个编码向量的输出。

![[公式]](https://www.zhihu.com/equation?tex=%5Ccolor%7Bdarkgreen%7D%7B%5Cmathbf%7Bx%7D_%5Ctext%7Bclass%7D%7D) 是一个可学习的嵌入向量，它的意义说通俗一点为：寻找其他9个输入向量对应的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bimage%7D) 的类别。

代码为：

```text
# dim=1024
self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

# forward前向代码
# 变成(b,64,1024)
cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
# 跟前面的分块进行concat
# 额外追加token，变成b,65,1024
x = torch.cat((cls_tokens, x), dim=1)
```

#### **Positional Encoding**

按照Transformer的位置编码的习惯，这个工作也使用了位置编码。**引入了一个 Positional encoding** ![[公式]](https://www.zhihu.com/equation?tex=%5Ccolor%7Bpurple%7D%7B%5Cmathbf%7BE%7D_%7Bpos%7D%7D) **来加入序列的位置信息**，同样在这里也引入了pos_embedding，是**用一个可训练的变量**。

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bz%7D_0+%3D+%5B+%5Cmathbf%7Bx%7D_%5Ctext%7Bclass%7D%3B+%5C%2C+%5Ccolor%7Bblack%7D%7B%5Cmathbf%7Bx%7D%5E1_p+%5Cmathbf%7BE%7D%3B+%5C%2C+%5Cmathbf%7Bx%7D%5E2_p+%5Cmathbf%7BE%7D%3B+%5Ccdots%3B+%5C%2C+%5Cmathbf%7Bx%7D%5E%7BN%7D_p+%5Cmathbf%7BE%7D+%7D%5D+%2B+%5Ccolor%7Bpurple%7D%7B%5Cmathbf%7BE%7D_%7Bpos%7D%7D+%5Ctag%7B5.2%7D)

没有采用原版Transformer的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bsincos%7D) 编码，而是直接设置为可学习的Positional Encoding，效果差不多。对训练好的pos_embedding进行可视化，如下图所示。我们发现，**位置越接近，往往具有更相似的位置编码。此外，出现了行列结构；同一行/列中的patch具有相似的位置编码。**

![img](https://pic4.zhimg.com/80/v2-16e7ed41532b112607ec4a47e2dba7bb_1440w.jpg)图：ViT的可学习的Positional Encoding

```python3
# num_patches=64，dim=1024,+1是因为多了一个cls开启解码标志
self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
```

**Transformer Encoder的前向过程**

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+++++%5Cmathbf%7Bz%7D_0+%26%3D+%5B+%5Cmathbf%7Bx%7D_%5Ctext%7Bclass%7D%3B+%5C%2C+%5Cmathbf%7Bx%7D%5E1_p+%5Cmathbf%7BE%7D%3B+%5C%2C+%5Cmathbf%7Bx%7D%5E2_p+%5Cmathbf%7BE%7D%3B+%5Ccdots%3B+%5C%2C+%5Cmathbf%7Bx%7D%5E%7BN%7D_p+%5Cmathbf%7BE%7D+%5D+%2B+%5Cmathbf%7BE%7D_%7Bpos%7D%2C+++++%26%26+%5Cmathbf%7BE%7D+%5Cin+%5Cmathbb%7BR%7D%5E%7B%28P%5E2+%5Ccdot+C%29+%5Ctimes+D%7D%2C%5C%2C+%5Cmathbf%7BE%7D_%7Bpos%7D++%5Cin+%5Cmathbb%7BR%7D%5E%7B%28N+%2B+1%29+%5Ctimes+D%7D+%5Clabel%7Beq%3Aembedding%7D+%5C%5C%5C%5C+++++%5Cmathbf%7Bz%5E%5Cprime%7D_%5Cell+%26%3D+%5Ccolor%7Bpurple%7D%7B%5Ctext%7BMSA%7D%7D%28%5Ccolor%7Bpurple%7D%7B%5Ctext%7BLN%7D%7D%28%5Cmathbf%7Bz%7D_%7B%5Cell-1%7D%29%29+%2B+%5Cmathbf%7Bz%7D_%7B%5Cell-1%7D%2C+%26%26+%5Cell%3D1%5Cldots+L+%5Clabel%7Beq%3Amsa_apply%7D+%5C%5C%5C%5C+++++%5Cmathbf%7Bz%7D_%5Cell+%26%3D+%5Ccolor%7Bteal%7D%7B%5Ctext%7BMLP%7D%7D%28%5Ccolor%7Bteal%7D%7B%5Ctext%7BLN%7D%7D%28%5Cmathbf%7Bz%5E%5Cprime%7D_%7B%5Cell%7D%29%29+%2B+%5Cmathbf%7Bz%5E%5Cprime%7D_%7B%5Cell%7D%2C+%26%26+%5Cell%3D1%5Cldots+L++%5Clabel%7Beq%3Amlp_apply%7D+%5C%5C%5C%5C+++++%5Cmathbf%7By%7D+%26%3D+%5Ctext%7BLN%7D%28%5Cmathbf%7Bz%7D_L%5E0%29+%5Clabel%7Beq%3Afinal_rep%7D+%5Cend%7Balign%7D+%5Ctag%7B5.3%7D)

其中，第1个式子为上面讲到的Patch Embedding和Positional Encoding的过程。

第2个式子为Transformer Encoder的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ccolor%7Bpurple%7D%7B%5Ctext%7BMulti-head+Self-attention%2C+Add+and+Norm%7D%7D) 的过程，重复 ![[公式]](https://www.zhihu.com/equation?tex=L) 次。

第3个式子为Transformer Encoder的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ccolor%7Bteal%7D%7B%5Ctext%7BFeed+Forward+Network%2C+Add+and+Norm%7D%7D) 的过程，重复 ![[公式]](https://www.zhihu.com/equation?tex=L) 次。

作者采用的是没有任何改动的transformer。
最后是一个 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BMLP%7D) 的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BClassification+Head%7D) ，整个的结构只有这些，如下图所示，为了方便读者的理解，我把变量的维度变化过程标注在了图中。

![img](https://pic2.zhimg.com/80/v2-7439a17c2e9aa981c95d783a93cb8729_1440w.jpg)图：ViT整体结构

**训练方法：**

先在大数据集上预训练，再迁移到小数据集上面。做法是把ViT的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ccolor%7Bpurple%7D%7B%5Ctext%7Bprediction+head%7D%7D) 去掉，换成一个 ![[公式]](https://www.zhihu.com/equation?tex=D%5Ctimes+K) 的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ccolor%7Bpurple%7D%7B%5Ctext%7BFeed+Forward+Layer%7D%7D) 。其中 ![[公式]](https://www.zhihu.com/equation?tex=K) 为对应数据集的类别数。

当输入的图片是更大的shape时，patch size ![[公式]](https://www.zhihu.com/equation?tex=P) 保持不变，则 ![[公式]](https://www.zhihu.com/equation?tex=N%3DHW%2FP%5E2) 会增大。

ViT可以处理任意 ![[公式]](https://www.zhihu.com/equation?tex=N) 的输入，但是Positional Encoding是按照预训练的输入图片的尺寸设计的，所以输入图片变大之后，Positional Encoding需要根据它们在原始图像中的位置做2D插值。

#### 代码解读

首先是介绍使用方法：

**安装：**

```console
pip install vit-pytorch
```

**使用：**

```python3
import torch
from vit_pytorch import ViT

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 256, 256)
mask = torch.ones(1, 8, 8).bool() # optional mask, designating which patch to attend to

preds = v(img, mask = mask) # (1, 1000)
```

> **传入参数的意义：**
> **image_size：**输入图片大小。
> **patch_size：**论文中 patch size： ![[公式]](https://www.zhihu.com/equation?tex=P) 的大小。
> **num_classes：**数据集类别数。
> **dim：**Transformer的隐变量的维度。
> **depth：**Transformer的Encoder，Decoder的Layer数。
> **heads：**Multi-head Attention layer的head数。
> **mlp_dim：**MLP层的hidden dim。
> **dropout：**Dropout rate。
> **emb_dropout：**Embedding dropout rate。



> **定义残差，** ![[公式]](https://www.zhihu.com/equation?tex=%5Ccolor%7Bpurple%7D%7B%5Ctext%7BFeed+Forward+Layer%7D%7D) **等：**

```python
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
```



> Attention和Transformer，注释已标注在代码中：

```python
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
# b, 65, 1024, heads = 8
        b, n, _, h = *x.shape, self.heads

# self.to_qkv(x): b, 65, 64*8*3
# qkv: b, 65, 64*8
        qkv = self.to_qkv(x).chunk(3, dim = -1)

# b, 65, 64, 8
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

# dots:b, 65, 64, 64
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

# attn:b, 65, 64, 64
        attn = dots.softmax(dim=-1)

# 使用einsum表示矩阵乘法：
# out:b, 65, 64, 8
        out = torch.einsum('bhij,bhjd->bhid', attn, v)

# out:b, 65, 64*8
        out = rearrange(out, 'b h n d -> b n (h d)')

# out:b, 65, 1024
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x
```



> ViT整体：

```python
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, mask = None):
        p = self.patch_size

# 图片分块
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)

# 降维(b,N,d)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

# 多一个可学习的x_class，与输入concat在一起，一起输入Transformer的Encoder。(b,1,d)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)

# Positional Encoding：(b,N+1,d)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

# Transformer的输入维度x的shape是：(b,N+1,d)
        x = self.transformer(x, mask)

# (b,1,d)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
# (b,1,num_class)
```
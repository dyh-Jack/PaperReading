#### Prioritized Experience Replay

PER提出的motivation是随机ER可能会导致具有学习价值的稀有的经验在Replay过程中无法被充分利用

![image-20220322170619359](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220322170619359.png)

蓝色曲线代表采用了某一种oracle使得每次都挑选能够最大限度减少全局损失的转换，从而可以看出，蓝色曲线收敛远快于随机ER。

因此，我们采用一种优先ER，优先级的衡量依据TD-error，也就是$\delta$：

![image-20220322174325374](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220322174325374.png)

但是如果我们依据$\delta$的绝对大小进行排名，贪心地选取$\delta$较大的进行学习，可能会出现一些问题：

- 很可能一个 (s,a) 在第一次访问的时候 $\delta$ 很小，因此被排到后面，但是之后的网络调整让这个的$\delta$变得很大，然而我们很难再调整它了（必须把它前面的都调整完）
- 当r是随机的时候（也就是环境具有随机性），$\delta$的波动就很剧烈，这会让训练的过程变得很敏感。
- greedy的选择次序很可能让神经网络总是更新某一部分样本，也就是“经验的一个子集”，这可能陷入局部最优，也可能导致过拟合的发生（尤其是在函数近似的过程中）。

因此，我们提出两种新的确定优先度的方法：

- proportional prioritization

![image-20220322174723299](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220322174723299.png)

![image-20220322174733400](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220322174733400.png)

- rank-based prioritization

![image-20220322174754989](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220322174754989.png)

rank-based相对来说更加具有鲁棒性，因为其对于离群值并不敏感，异常数值的TD-error影响不大。不过实际实验两种方式效果相近。

而在代码设计中，我们采用SumTree的数据结构实现，基本思路如下：

![image-20220325152524771](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220325152524771.png)

```python
class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        
    """
    
    def __init__(
        self, 
        obs_dim: int,
        size: int, 
        batch_size: int = 32, 
        alpha: float = 0.6
    ):
        """Initialization."""
        assert alpha >= 0
        
        super(PrioritizedReplayBuffer, self).__init__(obs_dim, size, batch_size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store(
        self, 
        obs: np.ndarray, 
        act: int, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool
    ):
        """Store experience and priority."""
        super().store(obs, act, rew, next_obs, done)
        
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        
        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )
        
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight
```

最核心的部分也就是sample的方式

```python
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
```

基本思路是按照上图每个节点占据整体的不同范围，在取n个数时，先将总体均分为n个部分，在每一部分随机取一个数，然后找到这个数所对应的节点，将其sample出来即可。

由于PER引入了偏差，因此我们需要一种方式纠正这种偏差。常见的是使用重要性抽样权值来纠正偏差：

![image-20220325153152268](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220325153152268.png)

同时，为了保持稳定性，我们通常利用$1/max\  w_i$来进行归一化
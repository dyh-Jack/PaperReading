## A Review on Video Prediction

### 1. What is the problem

#### 1.1 Define the Problem

![image-20220206214008408](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220206214008408.png)

#### 1.2 Exploiting the Time Dimension of Videos

![image-20220206214246586](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220206214246586.png)

![image-20220206214357142](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220206214357142.png)

#### 1.3 Dealing with Stochasticity

![image-20220206214547879](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220206214547879.png)

#### 1.4 The Devil is in the Loss Function

![image-20220206214753909](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220206214753909.png)

**<u>对于不同损失函数带来的效果的理解：</u>**

![image-20220206215027422](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220206215027422.png)

![image-20220206215056393](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220206215056393.png)

对于损失函数带来的视频预测模糊化：

![image-20220206215151603](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220206215151603.png)

下面两个图片的内容阐述了对于处理模糊化的一些常见思路：

![image-20220206215519199](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220206215519199.png)

![image-20220206215529974](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220206215529974.png)

大意如下：

![image-20220206215542737](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220206215542737.png)

### 2. Backbone Deep Learning Architectures

#### 2.1 Convolutional Models

基本问题：

![image-20220206215742335](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220206215742335.png)

对于解决卷积核感受野的方法：

![image-20220206215837875](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220206215837875.png)

大意如下：

![image-20220206215850194](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220206215850194.png)

引入3D卷积用来更好的解决卷积在时间尺度上的不足：

![image-20220206220028825](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220206220028825.png)

####  2.2 Recurrent Models

RNN表现较为成功，是出色的序列到序列的模型，但受到梯度消失和梯度爆炸问题的影响，普通的RNN在处理较长时间序列时遇到了困难，后续改进的RNN缓解了这一问题。

#### 2.3  Generative Models

相关知识亟待补充。。。。。。

### 3. Datasets

#### 3.1 Action and Human Pose Recognition Datasets

#### 3.2 Driving and Urban Scene Understanding Datasets

#### 3.3 Object and Video Classification Datasets

#### 3.4 Video Prediction Datasets

![image-20220206221344537](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220206221344537.png)

### 4. Video Prediction Methods

![image-20220206222525551](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220206222525551.png)

#### 4.1 Direct Pixel Synthesis

##### 4.1.1 Inspired on adversarial training

相关知识亟待补充。。。。。。

##### 4.1.2 Bidirectional flow

即进行回顾式的预测，同时关注向前和向后的预测。

##### 4.1.3 Exploiting 3D convolutions

![image-20220206223416172](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220206223416172.png)

![image-20220206223504541](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220206223504541.png)

![image-20220206223646245](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220206223646245.png)

**从原始像素中提取信息会导致在长期预测中误差的指数级增长。**

#### 4.2 Using Explicit Transformations（not understood）

##### 4.2.1 Vector-based Resampling

##### 4.2.2 Kernel-based Resampling

#### 4.3 Explicit Motion from Content Separation

侧重于将运动与内容相分离。

MotionRNN即是此思路，对运动进行了显式的建模。

#### 4.4 Conditioned on Extra Variables

引入多变量的数据，通过图片中的其他信息辅助视频预测

#### 4.5 In the High-level Feature Space

##### 4.5.1 Semantic Segmentation

##### 4.5.2 Instance Segmentation

##### 4.5.3 Other High-level Spaces

#### 4.6 Incorporating Uncertainty

### 5. Performance Evaluation

#### 5.1 Metrics and Evaluation Protocols

![image-20220207221629252](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220207221629252.png)

#### 5.2 Results

##### 5.2.1 Results on Probabilistic Approaches

##### 5.2.2 Results on the High-level Prediction Space

### 6. Dicussion

![image-20220207222349322](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220207222349322.png)

![image-20220207222543353](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220207222543353.png)

大意如下：

![image-20220207222603186](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220207222603186.png)

#### 6.1 Research Challenges

![image-20220207222803988](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220207222803988.png)

大意如下：

![image-20220207222827791](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220207222827791.png)

#### 6.2 Future Directions

![image-20220207223141573](C:\Users\dyh20200207\AppData\Roaming\Typora\typora-user-images\image-20220207223141573.png)
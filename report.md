# Report

王琛 计65 2016011360

## 实验设计

实验的主要原理可以使用等式 (1)概括:

​							$ \hat y = argmax_{y}P(y) \prod_{i=1}^{n}P(x_{i}|y)$          (1)

我将实验大致分成了三个部分：

- 读取每个邮件的标签，然后随机分成五折. 
- 训练，计算概率
- 一个贝叶斯分类器

一开始我准备实现一个独立于使用场景的NB， 但是 $P(y)$ 和 $P(x_{i}|y)$ 的计算都和使用情景有关，因此我实现了两个类 -- `BayesClassifier`, 一个抽象类定义了使用朴素贝叶斯时需要用到的函数的参数以及 `BayesSpamFilter` 实现了`BayesClassifier`中的抽象方法。

实验代码结构如下 (`src`文件夹)：

- `config.py`：定义了实验中需要用到的参数，比如数据的路径，哪里存放训练结果等等。比较重要的是随机种子的设置，实验中有两个地方用到了随机数种子。一个是划分5折时 (`parser.py`), 对应了`config.py`中的`fold_seed`，是一个5个元素的列表，表示issue1中5次随机采样。另外一个地方时选择验证集大小时 (`test.py`)，对应的是`train_seed=17`。
- `parser.py`: 
  - `get_label`: 读取文件 `label/index` 并且返回一个dict { 'email_path' : 'spam or ham' }
  - `divide_into_folds`: 使用 `get_label` 返回的dict然后将每个email划分到5折中的一个，并且将5折的结果保存到`../dataset`中。
- `train.py`:
  - `get_set`: 从5折数据中得到测试集和训练集
  - `train`: 计算训练集中spam和ham邮件的个数，计算每个词在spam和ham出现的次数，用于后面计算概率使用。参数 `train_per` 表示使用训练集的比例。

- `spamfilter.py`:
  - `class BayesClassifier`: 定义了 `cal_py`, `cal_p_xi_y`, `classify`的抽象类
  - `class BayesSpamFilter`: 
    - `cal_py`: 计算spam和ham邮件的概率
    - `cal_p_xi_y`: 计算等式(1)中的 $P(x_{i}|y)$ ，也就是当`y`是spam或者ham时词 $x_{i}$ 出现的概率
    - `split_email`: to be finished

- `test.py`：


## Issue Addressing

### Issue 1

划分五折数据时，五次采用的随机数种子分别是`fold_seed=[1999, 538, 2297, 154, 958]`，结果如下表所示：
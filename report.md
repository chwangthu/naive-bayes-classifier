# Report

王琛 计65 2016011360

## 实验设计

实验的主要原理可以使用等式 (1)概括:

​							$ \hat y = argmax_{y}P(y) \prod_{i=1}^{n}P(x_{i}|y)$          (1)

我将实验大致分成了三个部分：

- 读取每个邮件的标签，然后随机分成五折. 
- 训练，计算概率
- 一个贝叶斯分类起 

At first, I plan to implement a NB that is independent of usage scenarios. But it seems both $P(y)$ and $P(x_{i}|y)$ are relevant to specific conditions, therefore I create two classes -- `BayesClassifier`, which is an abstract class that defines the methods and arguments needed for all conditions, and `BayesSpamFilter` that implements abstract methods of `BayesClassifier` in spam filtering conditions.

The organization of source code (`src` folder) is listed as follows:

- `config.py`：Some parameters needed in the project, such as the path of original data, where to store training result etc. `random_seed` is used when dispatching folds (`parser.py`), and `train_seed` is used when selecting the size of training set (`test.py`).
- `parser.py`: 
  - `get_label`: read file `label/index` and return a dict like { 'email_path' : 'spam or ham' }
  - `divide_into_folds`: use the dict `get_label` returns and dispatch each email into a fold, write all of five folds into `../dataset`
- `test.py`:
  - `get_set`: get trainset and test set from 5 folds
  - `train`: count the number of spam and ham emails, as well as the times of word appearances. Use argument `train_per` to specify the percent of train set to use. 

- `spamfilter.py`:
  - `class BayesClassifier`: abstract class that defines `cal_py`, `cal_p_xi_y`, `classify`
  - `class BayesSpamFilter`: 
    - `cal_py`: calculate the probability of spam or ham emails.
    - `cal_p_xi_y`: calculate $P(x_{i}|y)$ in equation (1), that is the possibility of word $x_{i}$ appears when email type is y.
    - `split_email`: to be finished



## Issue Addressing

### Issue 1


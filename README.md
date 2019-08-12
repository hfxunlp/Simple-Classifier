# Simple Classifier
A simple feature based classifier developed with python 3.7.

## 数据
训练数据是train_data.txt，每一行第一个token是分类标签，其它token是特征。
## TFIDF模型统计
运行：
`python tfidftrain.py train_data.txt tfidf_model.txt`
可以生成一个tfidf的统计模型;
## 模型调优
运行：
`python train.py train_data.txt tfidf_model.txt trained_model train_epoch init_lr regularize_weight`
可以用生成式算法调优TFIDF生成的模型，train_epoch是训练周期，init_lr是初始学习速率，regularize_weight是L2 regularizer的权重（一般0.0禁用即可，可以直接不传此参数）。模型会被保存为trained_model_$epoch_$loss.txt。
## 预测
进行测试：
`python predict.py text_data.txt trained_model_epoch_loss.txt test_result.txt bias_weight`
text_data.txt是测试数据，一行一条数据的所有特征。trained_model.txt是之前生成的任意一个模型文件。test_result.txt是测试结果，对应一行一个类别标签。bias_weight是用来控制每个类别加的bias的权重，默认1.0，一般不传即可，除非作为超参数调整TFIDF模型。

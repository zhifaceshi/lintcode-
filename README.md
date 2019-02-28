# lintcode-电影评论识别

[lintcode电影评论识别](https://www.lintcode.com/ai/movie-review-recognition/overview)  
:laughing:
***
+ 使用Bilstm神经网络
+ 使用CNN神经网络
+ 使用glove词向量

***
结果    
模型|训练轮数|分数     
:----:|:-----:|:----:  
BiLSTM|10|99.263%      
CNN|30|99.919%     
Bert-base|10|65.546%   
transformer-4layers|100|99.980%  
***
+ 用的Bert预训练模型，但是效果差了很多。  
同时训练的时间也很漫长。
+ 我想实验一下transformer到底有没有那么神奇，于是使用4层的transformer，并且修改了一些参数使参数量更小，但是用3e-3的学习率期初没有下降，最后调到1e-4才使损失下降。
+ 尝试使用adabound参数优化  [中国学霸本科生提出AI新算法：速度比肩Adam，性能媲美SGD](https://baijiahao.baidu.com/s?id=1626597958173084746&wfr=spider&for=pc)

+ BERT模型占用很大的GPU空间，当batchsize=32,12G显存装不下，建议不开启GPU模式进行训练
***

# LintCode句子情感分类
[句子情感分类](https://www.lintcode.com/ai/UMICH_Sentiment_Analysis/overview)



# lintcode-电影评论识别

[lintcode电影评论识别](https://www.lintcode.com/ai/movie-review-recognition/overview)  
:)
***
+ 使用Bilstm神经网络
+ 使用CNN神经网络
+ 使用glove词向量

***
结果    
模型|训练轮数|分数    
-|-|-    
BiLSTM|10|99.263%     
CNN|30|99.919%    
Bert-base|10|65.546%  

用的Bert预训练模型，但是效果差了很多。  
同时训练的时间也很漫长。
***
+ BERT模型占用很大的GPU空间，当batchsize=32,12G显存装不下，建议不开启GPU模式进行训练



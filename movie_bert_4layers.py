from data.data import NonBertData
from model.bert_4layers import MyBert
from torchtext import data
from Verify.Train import Trainer
from Verify.Test import Tester
import torch
import pandas as pd
from Config import Config
if __name__ == '__main__':
    # train_pth, train_fields, test_pth, test_fields, TEXT, LABEL, x_var, y_vars, x_test_var, y_test_vars,
    Config.train_batchsize=45
    train_pth = './data/movie_train.tsv'
    test_pth = './data/movie_test.tsv'
    TEXT = data.Field(batch_first=True)
    LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True)
    train_fields = [('id', None), ('sentiment', LABEL), ('review', TEXT)]
    test_fields = [('id', None), ('review', TEXT)]
    x_var = 'review'
    y_vars = ['sentiment']
    x_test_var = 'review'
    
    nonbertdata = NonBertData(train_pth, train_fields, test_pth, test_fields, TEXT, LABEL, x_var, y_vars, x_test_var,
                              None)
    train_iter, test_iter = nonbertdata.getdata()
    
    model = MyBert(len(TEXT.vocab),TEXT.vocab.vectors)
    # a=list(model.parameters())
    # optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    ######################################################
    import adabound
    optimizer=adabound.AdaBound(model.parameters(), lr=2e-5,final_lr=0.1)
    #######################################################
    train_test = True
    
    modelname = "./models_pkl/Bert_4layers_MODEL.pkl"
    if train_test:
        trainer = Trainer(model, train_iter, optimizer, modelname, use_GPU=True)
        trainer.train(1000, pretrain_pth=modelname, padding_idx=1)
    else:
        tester = Tester(model, test_iter, use_GPU=True)
        df = tester.test(modelname, padding_idx=1)
        data = pd.read_csv(test_pth, header=0, delimiter='\t')
        newdataframe = pd.merge(data, df, left_index=True, right_index=True)
        retdataframe = newdataframe[['id', 0]]
        retdataframe.columns = ['id', 'sentiment']
        retdataframe.to_csv('./tmp/Bert_4layers__movie.csv', index=None)



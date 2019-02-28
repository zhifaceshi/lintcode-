from data.data import NonBertData
from model.CNN import CnnModel
from torchtext import data
from Verify.Train import Trainer
from Verify.Test import Tester
import torch
import pandas as pd

if __name__ == '__main__':
    # train_pth, train_fields, test_pth, test_fields, TEXT, LABEL, x_var, y_vars, x_test_var, y_test_vars,
    train_pth = './data/emotion_train.csv'
    test_pth = './data/emotion_train.csv'
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
    # print(TEXT.vocab.stoi['watched'])
    # print(TEXT.vocab.stoi['it'])
    model = CnnModel(len(TEXT.vocab), 100, 2, vector=TEXT.vocab.vectors, padding_idx=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    
    train_test = False
    modelname = "./models_pkl/CNN_MODEL.pkl"
    if train_test:
        trainer = Trainer(model, train_iter, optimizer, modelname, use_GPU=True)
        trainer.train(30, pretrain_pth=modelname, padding_idx=1)
    else:
        tester = Tester(model, test_iter, use_GPU=True)
        df = tester.test(modelname, padding_idx=1)
        data = pd.read_csv(test_pth, header=0, delimiter='\t')
        newdataframe = pd.merge(data, df, left_index=True, right_index=True)
        retdataframe = newdataframe[['id', 0]]
        retdataframe.columns = ['id', 'sentiment']
        retdataframe.to_csv('./tmp/CNN_movie.csv', index=None)




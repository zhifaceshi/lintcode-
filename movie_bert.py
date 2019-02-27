from data.data import BertData
from pytorch_pretrained_bert import BertTokenizer
from model.BERT_based import MyBert
from torchtext import data
from Verify.Train import Trainer
from Verify.Test import Tester
import torch
import pandas as pd
from Config import Config
if __name__ == '__main__':
    Config.train_batchsize=160
    # train_pth, train_fields, test_pth, test_fields, TEXT, LABEL, x_var, y_vars, x_test_var, y_test_vars,
    tokenizer = BertTokenizer.from_pretrained(Config.tokenizer_file)
    train_pth = './data/movie_train.tsv'
    test_pth = './data/movie_test.tsv'
    TEXT = data.Field(batch_first=True,tokenize=tokenizer.tokenize)
    LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True)
    train_fields = [('id', None), ('sentiment', LABEL), ('review', TEXT)]
    test_fields = [('id', None), ('review', TEXT)]
    x_var = 'review'
    y_vars = ['sentiment']
    x_test_var = 'review'

    print("loading model...")
    model = MyBert()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-3)
    
    print('loading data...')
    bertdata = BertData(train_pth, train_fields, test_pth, test_fields, TEXT, LABEL, x_var, y_vars, x_test_var,None,
                              tokenizer)
    train_iter, test_iter = bertdata.getdata()
    
    train_test = False
    
    modelname = "./models_pkl/BERT_MODEL.pkl"
    if train_test:
        
        trainer = Trainer(model, train_iter, optimizer, modelname, use_GPU=False)
        trainer.train(10, pretrain_pth=modelname, padding_idx=1)
    else:
        tester = Tester(model, test_iter, use_GPU=True)
        df = tester.test(modelname, padding_idx=1)
        data = pd.read_csv(test_pth, header=0, delimiter='\t')
        newdataframe = pd.merge(data, df, left_index=True, right_index=True)
        retdataframe = newdataframe[['id', 0]]
        retdataframe.columns = ['id', 'sentiment']
        retdataframe.to_csv('./tmp/BERT_movie.csv', index=None)



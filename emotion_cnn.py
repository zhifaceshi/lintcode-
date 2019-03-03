from data.data import NonBertData
from model.CNN import CnnModel
from torchtext import data
from Verify.Train import Trainer
from Verify.Test import Tester
import torch
import pandas as pd
from Config import Config
from torchtext.data import Field
class myField(Field):
    def pad(self, minibatch):
        """
        因为原本的Field，没有min_length的功能，我们添加进去
        """
        minibatch = list(minibatch)
        if not self.sequential:
            return minibatch
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            #######################################################
            #修改的地方
            max_len = max(self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2,max(len(x) for x in minibatch))
            ########################################################
        padded, lengths = [], []
        for x in minibatch:
            if self.pad_first:
                padded.append(
                    [self.pad_token] * max(0, max_len - len(x)) +
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]))
            else:
                padded.append(
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]) +
                    [self.pad_token] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return (padded, lengths)
        return padded
if __name__ == '__main__':
    # train_pth, train_fields, test_pth, test_fields, TEXT, LABEL, x_var, y_vars, x_test_var, y_test_vars,
    Config.train_batchsize=50
    train_pth = './data/emotion_train.csv'
    test_pth = './data/emotion_test.csv'
    TEXT = myField(batch_first=True,fix_length=10)
    LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True)
    train_fields = [ ('sentence', TEXT), ('label', LABEL)]
    test_fields = [ ('sentence', TEXT)]
    x_var = 'sentence'
    y_vars = ['label']
    x_test_var = 'sentence'
    
    nonbertdata = NonBertData(train_pth, train_fields, test_pth, test_fields, TEXT, LABEL, x_var, y_vars, x_test_var,
                              None)
    train_iter, test_iter = nonbertdata.getdata()
    # print(TEXT.vocab.stoi['watched'])
    # print(TEXT.vocab.stoi['it'])
    model = CnnModel(len(TEXT.vocab), 100, 2, vector=TEXT.vocab.vectors, padding_idx=1)
    import adabound
    optimizer=adabound.AdaBound(model.parameters(), lr=2e-4,final_lr=0.1)
    
    train_test = False
    modelname = "./models_pkl/CNN_EMOTION.pkl"
    use_GPU=False
    if train_test:
        trainer = Trainer(model, train_iter, optimizer, modelname, use_GPU=use_GPU)
        trainer.train(100, pretrain_pth=modelname, padding_idx=1)
    else:
        tester = Tester(model, test_iter, use_GPU=use_GPU)
        df = tester.test(modelname, padding_idx=1)
        data = pd.read_csv(test_pth, header=0, delimiter='\t')
        newdataframe = pd.merge(data, df, left_index=True, right_index=True)
        retdataframe = newdataframe[['sentence', 0]]
        retdataframe.columns = ['sentence', 'label']
        retdataframe.to_csv('./tmp/CNN_emotion.csv', index=None)




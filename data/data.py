from torchtext import data
import torch
from collections import defaultdict
from Config import Config
import json
from torchtext.vocab import Vectors

class DATA():
    def __init__(self,train_pth,train_fields,test_pth,test_fields,TEXT,LABEL,x_var,y_vars,x_test_var,y_test_vars,tokenizer=None):
        self.train_pth=train_pth
        self.fields=train_fields
        self.test_pth=test_pth
        self.test_fields=test_fields
        
        self.TEXT=TEXT
        self.LABEL=LABEL
        self.x_var=x_var
        self.y_vars=y_vars
        self.x_test_var=x_test_var
        self.y_test_vars=y_test_vars
        self.tokenizer = tokenizer
        
    def getdata(self):
        csvortsv = self.train_pth.split('.')[-1]
        train_data = data.TabularDataset(path=self.train_pth, fields=self.fields, format=csvortsv, skip_header=True)
        self.build_word(train_data,self.tokenizer)
        
        train_iter = data.BucketIterator(train_data, batch_size=Config.train_batchsize,sort_within_batch=True,sort_key=lambda x:getattr(x,self.x_var))
        train_iter=BatchWrapper(train_iter,self.x_var,self.y_vars)
        
        ######next is test iterator
        csvortsv = self.test_pth.split('.')[-1]
        test_data = data.TabularDataset(path=self.test_pth, fields=self.test_fields, format=csvortsv,skip_header=True)

        test_iter = data.Iterator(test_data, batch_size=1,train=False,sort=False,sort_within_batch=False)

        test_iter = BatchWrapper(test_iter, self.x_test_var)

        return train_iter,test_iter


    def build_word(self,train_data,tokenizer):
        raise NotImplementedError
        
class BertData(DATA):
    def __init__(self,train_pth,train_fields,test_pth,test_fields,TEXT,LABEL,x_var,y_vars,x_test_var,y_test_vars,tokenizer=None):
        super(BertData, self).__init__(train_pth,train_fields,test_pth,test_fields,TEXT,LABEL,x_var,y_vars,x_test_var,y_test_vars,tokenizer)
        
    def build_word(self,train_data,tokenizer):
        
        self.TEXT.build_vocab(train_data[self.x_var])
        self.TEXT.vocab.itos=self.tokenizer.ids_to_tokens
        a = defaultdict(lambda: self.tokenizer.vocab['[UNK]'])
        a.update({i: tok for i, tok in (self.tokenizer.vocab).items()})
        self.TEXT.vocab.stoi = a
     
    
class NonBertData(DATA):
    def __init__(self,train_pth,train_fields,test_pth,test_fields,TEXT,LABEL,x_var,y_vars,x_test_var,y_test_vars,tokenizer=None):
        super(NonBertData, self).__init__(train_pth,train_fields,test_pth,test_fields,TEXT,LABEL,x_var,y_vars,x_test_var,y_test_vars,tokenizer)

    def build_word(self, train_data, tokenizer):
        vector = Vectors(name=Config.vector_pth)
        self.TEXT.build_vocab(train_data,vectors=vector)
 
        

class BatchWrapper():
    def __init__(self, d1, x_var, y_vars=None):
        self.d1 = d1
        self.x_var = x_var
        
        self.y_vars = y_vars
    
    def __iter__(self):
        for batch in self.d1:
            x = getattr(batch, self.x_var)
            
            
            x = x[:, :512]
            if self.y_vars is None:
                yield x
            else:
                assert isinstance(self.y_vars, list)
                if self.y_vars is not None:
                    temp = [getattr(batch, feat).unsqueeze(1) for feat in self.y_vars]
                    y = torch.cat(temp, dim=1).float()
                else:
                    y = torch.zeros((1))
                yield (x, y.squeeze(1).long())
    
    def __len__(self):
        return len(self.d1)
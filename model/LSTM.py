import torch
import torch.nn as nn
from model.BaseModel import BaseModel

class LSTM(nn.Module):
    def __init__(self,vocab_size,embedding_dim,target_size,vector=None,padding_idx=0):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,embedding_dim,padding_idx=padding_idx)
        self.lstm=nn.LSTM(
            input_size=embedding_dim,
            hidden_size=768//2,
            num_layers=2,
            bias=True,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        self.out=nn.Linear(768*2,target_size)
        if vector is not None:
            self.embed.weight.data.copy_(vector)
    def forward(self, x,mask):
        x=self.embed(x)
        
        mask=mask.sum(1).int()
        x=nn.utils.rnn.pack_padded_sequence(x,mask,batch_first=True)
        h,_=self.lstm(x)
        h,_=nn.utils.rnn.pad_packed_sequence(h,batch_first=True)
        # a=h[-1,:,:]
        hidden=torch.cat((h[:,-1,:],h[:,-1,:]),dim=1)
        h=self.out(hidden)
        return h
    
class LstmModel(BaseModel):
    def __init__(self,vocab_size,embedding_dim,target_size,vector=None,padding_idx=1):
        super(LstmModel, self).__init__()
        self.model=LSTM(vocab_size,embedding_dim,target_size,vector=vector,padding_idx=padding_idx)
    def forward(self, x,mask=None,labels=None):
        if labels is None:
            output=self.model(x,mask)
            output=torch.argmax(output,dim=1)
            return output
        else:
            output = self.model(x, mask)
            loss=self.loss_func(output,labels)
            return loss
        
        
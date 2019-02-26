import torch
import torch.nn as nn
from model.BaseModel import BaseModel
import torch.nn.functional as F

class Cnn(nn.Module):
    def __init__(self,vocab_size,embedding_dim,n_filters,filter_sizes,target,vector=None,padding_idx=0):
        super(Cnn, self).__init__()
        self.embedding=torch.nn.Embedding(vocab_size,embedding_dim,padding_idx=padding_idx)
        self.convs=torch.nn.ModuleList([torch.nn.Conv2d(in_channels=1,out_channels=n_filters,kernel_size=(fs,embedding_dim)) for fs in filter_sizes] )
        self.fc=nn.Linear(len(filter_sizes)*n_filters,target)
        self.drop=nn.Dropout(0.1)
        if vector is not None:
            self.embedding.weight.data.copy_(vector)
    def forward(self,x):
        embedded=self.embedding(x)
        embedded=embedded.unsqueeze(1)
        conved=[F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled=[F.max_pool1d(conv,conv.shape[2]).squeeze(2) for conv in conved]
        cat=self.drop(torch.cat(pooled,dim=1))
        return self.fc(cat)

class CnnModel(BaseModel):
    def __init__(self,vocab_size,embedding_dim,target_size=2,vector=None,padding_idx=0):
        super(CnnModel, self).__init__()
        n_filters, filter_sizes=100,[2,3,4,5]
        self.model=Cnn(vocab_size,embedding_dim,n_filters,filter_sizes,target_size,vector=vector)

    def forward(self, x,mask=None,labels=None):
        if labels  is None:
            output = self.model(x)
            output = torch.argmax(output, dim=1)
            return output
        else:
            output = self.model(x)
            loss = self.loss_func(output, labels)
            return loss
from pytorch_pretrained_bert import BertModel

import torch
import torch.nn as nn
from model.BaseModel import BaseModel
import torch.nn.functional as F
from Config import Config

class Bert_model(nn.Module):
    def __init__(self):
        super(Bert_model, self).__init__()

        self.bert=BertModel.from_pretrained(Config.bert_model_dir)
        for p in self.bert.parameters():
            p.requires_grad=False
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768,2)
 

    def forward(self, input_ids, attention_mask=None, labels=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
    
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        _, pooled_output = self.bert(flat_input_ids, None, flat_attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, 2)
    
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss
        else:
            return reshaped_logits
       
            
class MyBert(BaseModel):
    def __init__(self):
        super(MyBert, self).__init__()
        self.model=Bert_model()
    def forward(self, x,mask=None,labels=None):
        if labels is None:
            output=self.model(x,mask)
            output=torch.argmax(output,dim=1)
            return output
        else:
            output = self.model(x, mask)
            loss=self.loss_func(output,labels)
            return loss
    
    
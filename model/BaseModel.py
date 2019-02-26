import torch
class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.model=None
        self.loss_func=torch.nn.CrossEntropyLoss()
    def forward(self, x,mask=None,labels=None):
        pass
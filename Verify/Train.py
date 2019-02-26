
import os
import torch
from tqdm import tqdm

class Trainer():
    def __init__(self,model,iterator,optimizer,name,use_GPU=True):
        self.model=model
        self.iterator=iterator
        self.optimizer=optimizer
        self.name=name
        self.use_GPU=use_GPU
    def train(self,epochs,pretrain_pth=None,padding_idx=0):
        print("Training...")
        if pretrain_pth!=None and os.path.exists(pretrain_pth):
            print("Loading previous model...")
            self.model.load_state_dict(torch.load(pretrain_pth))
        self.model.train()
        pre_loss=1e10
        if self.use_GPU:
            print("Now is using GPU....")
            self.model.cuda()
        for i in range(epochs):
            bar=tqdm(self.iterator)
            epoch_loss=0
            N=0
            for batch in bar:
                self.optimizer.zero_grad()
                word,target=batch
                mask=word.ge(padding_idx).long()
                if self.use_GPU:
                    word=word.cuda()
                    target=target.cuda()
                
                loss=self.model(word,mask,target)
                loss=loss.sum()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                N+=len(batch)
                bar.set_description('Epoch:{}/{},mean loss is {}'.format(i+1, epochs, loss.item()/len(batch)))
            if pre_loss > epoch_loss / N:
                print("Now is saving model...")
                torch.save(self.model.state_dict(), self.name)
                pre_loss = epoch_loss / len(self.iterator)
            print("This epoch mean loss is {}\n".format(epoch_loss / N))

                
            
        
        
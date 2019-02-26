
import torch
from tqdm import tqdm
import pandas as pd
class Tester():
    def __init__(self,model,iterator,use_GPU=True):
        self.model = model
        self.iterator = iterator

        self.use_GPU = use_GPU
        
    def test(self,model_pth,padding_idx=0):
        print('Testing...')
        print('loading saved model...')
        self.model.load_state_dict(torch.load(model_pth))
        self.model.eval()
        
        if self.use_GPU:
            print("Now is using GPU....")
            self.model.cuda()
        bar = tqdm(self.iterator)
        ret=[]
        for batch in bar:
            # print('ddd')
            with torch.no_grad():
                word = batch
                mask = word.ge(padding_idx).long()
                assert len(word)==1
                if self.use_GPU:
                    word = word.cuda()
                    mask=mask.cuda()
                prediction=self.model(word,mask)
                
                ret.append(prediction.item())
        df = pd.DataFrame(ret)
        return df

                
                
                
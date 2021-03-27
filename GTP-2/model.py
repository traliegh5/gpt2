import torch
import torch.nn as nn
from transformers import   GPT2LMHeadModel

class GPT2_Transformer(nn.Module):
    def __init__(self):
        super(GPT2_Transformer, self).__init__()
        '''
        Load the pre-trained GPT2 Language Model Head Model

        '''
        
        


        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        pass

    def forward(self, inputs,masks):
       
        output=self.model(inputs,labels=inputs,attention_mask=masks)

        return output
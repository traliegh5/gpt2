from comet_ml import Experiment
import torch
import torch.nn
import argparse
from transformers import *
from model import *
from transformer import *
from preprocess import *
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


tokenizer= GPT2Tokenizer.from_pretrained('gpt2')

sentence="hello, my name is charlie"
sentenceTwo="hello my name is char"
sentence3="hello my name is matthias and I came here to kick ass and chew bubble gum"

sentences=[sentence,sentenceTwo,sentence3]
tokens=tokenizer(sentence,return_tensor="pt")
tokens2=tokenizer(sentenceTwo,return_tensor="pt")
tokens3=tokenizer(sentence3,return_tensor="pt")
seqs=[tokens['input_ids'],tokens2['input_ids'],tokens3['input_ids']]
tense=[]
masks=[]
window_size=7
start=tokenizer.bos_token_id
for sequence in sentences:
  
    token=tokenizer(sequence,return_tensor="pt")['input_ids']
    token=token[:window_size]
    token.insert(0,start)
    token.append(start)
    
    tense.append(torch.as_tensor(token))
    temp=torch.ones(window_size+2)
    temp[len(token):]=0.0
    masks.append(temp)
masks=torch.stack(masks)


seqs=pad_sequence(tense,batch_first=True)

print(seqs,masks)

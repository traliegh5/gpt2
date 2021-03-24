import torch
import torch.nn as nn
import math

class Transformer(nn.Module):
    def __init__(self, vocab_size,embedding_size,window_size,device):
        super(Transformer, self).__init__()
        self.device=device
        self.vocab_size=vocab_size
        self.embedding_size=embedding_size
        self.window_size=window_size
        self.soft=nn.Softmax(dim=0)
        self.num_heads=3
        
        self.emb=nn.Embedding(self.vocab_size,self.embedding_size)
        self.posemb=self.build_pos_emb()
        self.dim_reduc=nn.Linear(6*self.embedding_size,self.vocab_size)
        mask=torch.zeros([self.window_size,self.window_size]).to(self.device)
        for i in range(0,self.window_size):
            for j in range(0,self.window_size):
                if i>j:
                    mask[i][j]=float('-inf')
        self.mask=mask
        self.encoder_list=nn.ModuleList()
        for i in range(6):
            self.encoder_list.append(encoderLayer(self.embedding_size,self.mask,self.num_heads))
        pass
    
    def build_pos_emb(self):
        posEmb=torch.zeros(self.window_size,self.embedding_size).to(self.device)
        for pos in range(self.window_size):
            for i in range(self.embedding_size):
                if i%2==0:
                    denom=10000**(2*i/self.embedding_size)
                    posEmb[pos][i]=math.sin(pos/denom)
                else:
                    denom=10000**(2*i/self.embedding_size)
                    posEmb[pos][i]=math.cos(pos/denom)

        return posEmb
    def forward(self, inputs):
        
        embInputs=self.emb(inputs)
        
        stop=inputs.shape[1]
        posEmb=self.posemb[stop-1,:]
        modelIn=embInputs+posEmb
        output=[]
        for encoder in self.encoder_list:
            output.append(encoder(modelIn))
        
        out=torch.cat(output,dim=2)
       
        out=self.dim_reduc(out)

        return out

class encoderLayer(nn.Module):
    def __init__(self,embedding_size,mask,num_heads):
        
        super().__init__()
        self.mask=mask
        
        self.embedding_size=embedding_size
        hidden_size=self.embedding_size
        self.L1=nn.Linear(self.embedding_size,hidden_size)
        self.L2=nn.Linear(hidden_size,hidden_size)
        self.L3=nn.Linear(hidden_size,self.embedding_size)
        self.norm=nn.LayerNorm(self.embedding_size)
        self.num_heads=num_heads
        self.attHeads=MultiHead(self.num_heads,self.embedding_size,self.mask)

        
        pass
    def forward(self,inputs):
        
        att=self.attHeads(inputs)
        
        attNew=self.norm(inputs+att)
        lin=self.L1(attNew)
        lin=self.L2(lin)
        lin=self.L3(lin)
        linNew=self.norm(lin+attNew)

        
        return linNew

class Attention(nn.Module):
    
    def __init__(self,embedding_size,mask):
        
        super().__init__()
        self.mask=mask
        self.embedding_size=embedding_size
       
        
        self.W_v=nn.Parameter(torch.rand([self.embedding_size,self.embedding_size]),requires_grad=True)
        self.W_q=nn.Parameter(torch.rand([self.embedding_size,self.embedding_size]),requires_grad=True)
        self.W_k=nn.Parameter(torch.rand([self.embedding_size,self.embedding_size]),requires_grad=True)
        # self.emb=nn.Embedding(self.vocab_size,self.embedding_size)
        self.soft=nn.Softmax(dim=1)
        
        pass
    def forward(self,inputs):
        
        # eee=self.emb(inputs)
        eee=inputs
        
        
        values=torch.matmul(eee,self.W_v)
        # values=eee*self.W_v
        queries=torch.matmul(eee,self.W_q)
        keys=torch.matmul(eee,self.W_k)
       
        attentionMat=torch.matmul(queries,torch.transpose(keys,1,2))
       
        attentionMat=attentionMat/math.sqrt(self.embedding_size)
        attentionMat=attentionMat+self.mask
        attentionMat=self.soft(attentionMat)

        attention=torch.matmul(attentionMat,values)

        return attention
class MultiHead(nn.Module):
    def __init__(self,num_heads,embedding_size,mask):
        
        super().__init__()
        self.mask=mask
        self.embedding_size=embedding_size
        
        
        self.num_heads=num_heads
        self.heads=nn.ModuleList()
        self.linear=nn.Linear(self.num_heads*self.embedding_size,self.embedding_size)
        for i in range(self.num_heads):
            self.heads.append(Attention(self.embedding_size,self.mask))
        pass
    def forward(self,inputs):
        output=[]
        for head in self.heads:
            output.append(head(inputs))
        out=torch.cat(output,dim=2)
        out=self.linear(out)
        
        return out 

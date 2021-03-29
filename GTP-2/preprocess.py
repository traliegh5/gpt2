from transformers import *
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

def load_dataset(train_file,test_file, window_size,tokenizer, batch_size,GPT):
    """
    :param fn: filename for the dataset
    :return: (torch.utils.data.DataLoader, torch.utils.data.DataLoader) for train and test
    :Comment: This preprocess step is different from the previous ones. In this assignment, we are interested in using a pre-trained model.
    So, we have to use the exact vocabulary the pre-trained model was trained with. We are using the GPT-2 model, so pass your data through
    the GPT-2 tokenizer to get the word ids. You don't need to create your own dictionary.
    """
    train_set=LanguageModel(train_file,window_size,tokenizer,GPT)
    test_set=LanguageModel(test_file,window_size,tokenizer,GPT)

    train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(test_set,batch_size=batch_size)
    
    return train_loader,test_loader




class LanguageModel(Dataset):
    def __init__(self, input_file,window_size, tokenizer,GPT):
        """
        Read and parse the translation dataset line by line. Make sure you
        separate them on tab to get the original sentence and the target
        sentence. You will need to adjust implementation details such as the
        vocabulary depending on whether the dataset is BPE or not.

        :param input_file: the data file pathname
        :param enc_seq_len: sequence length of encoder
        :param dec_seq_len: sequence length of decoder
        :param bpe: whether this is a Byte Pair Encoded dataset
        :param target: the tag for target language
        :param word2id: the word2id to append upon
        :param flip: whether to flip the ordering of the sentences in each line
        """
        # TODO: read the input file line by line and put the lines in a list.

        # TODO: split the whole file (including both training and validation
        # data) into words and create the corresponding vocab dictionary.

        # TODO: create inputs and labels for both training and validation data
        #       and make sure you pad your inputs.

        # Hint: remember to add start and pad to create inputs and labels
        
        
        self.lengths=[]
       
        self.tokenizer=tokenizer
        self.tense=[]
        self.masks=[]
        self.window_size=window_size-1

        self.start=tokenizer.bos_token_id
        print(self.start)
        if GPT:
            padding_value=0
        else:
            padding_value=50257
        
        with open(input_file,'r') as f:
            
            while True:
                line=f.readline()
                
                if not line:
                    break
                # cent=line.strip().split()
                # cent.insert(0,'START')
                token=self.tokenizer(line)['input_ids']
                token=token[:self.window_size]
                token.insert(0,self.start)
                token.append(self.start)
                
                self.tense.append(torch.as_tensor(token))
                temp=torch.ones(self.window_size+2)
                temp[len(token):]=0.0
                self.masks.append(temp)
                

                
                self.lengths.append(len(token)-1)
                
       
        self.tense=pad_sequence(self.tense,batch_first=True,padding_value=padding_value)
        print(max(self.lengths))
        self.masks=pad_sequence(self.masks,batch_first=True,padding_value=padding_value)
      
        
        
        
    
        
        
    
        
        
        pass

    def __len__(self):
        """
        len should return a the length of the dataset

        :return: an integer length of the dataset
        """
        # TODO: Override method to return length of dataset
        
        return self.tense.shape[0]

    def __getitem__(self, idx):
        """
        getitem should return a tuple or dictionary of the data at some index
        In this case, you should return your original and target sentence and
        anything else you may need in training/validation/testing.

        :param idx: the index for retrieval

        :return: tuple or dictionary of the data
        """
        # TODO: Override method to return the items in dataset
       
        item={"inputs":self.tense[idx,:],"lengths":self.lengths[idx],"labels":self.tense[idx,1:],"masks":self.masks[idx,:]}
        
        return item

    # def __len__(self):

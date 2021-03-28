from comet_ml import Experiment
import torch
import torch.nn
import argparse
from transformers import *
from model import *
from transformer import Transformer
from preprocess import *
from tqdm import tqdm
import math
import gc
from torch import nn, optim


hyper_params = {
     "batch_size": 100,
     "num_epochs": 1,
     "learning_rate": 0.001,
     "window_size":100,
     "embedding_size":512,
     "vocab_size":50257
 }
device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
experiment = Experiment(project_name="gtp-2")
experiment.log_parameters(hyper_params)

def train(experiment,model,hyper_params,train_loader):
    torch.cuda.empty_cache()
    
    loss_fn=nn.CrossEntropyLoss(ignore_index=hyper_params["vocab_size"])
    optimizer=optim.Adam(model.parameters(),lr=hyper_params["learning_rate"])
    model = model.train()
    total_loss = 0
    word_count = 0
    num_batches=0
    experiment.log_parameters({"num_heads":model.num_heads})
    with experiment.train():
        
        for i in range(hyper_params["num_epochs"]):
            accuracy_total=0
            total_units=0
            loss_tot=0
            batchnum=0
            for batch in tqdm(train_loader):
                inputs=batch["inputs"][:,0:-1].to(device)
                labels=batch["labels"][:,1:].to(device)
                lens=batch["lengths"].to(device)
                temp=torch.sum(lens)
                word_count+=temp
                total_units+=temp
                optimizer.zero_grad()
                #print(x.shape)
                preds = model(inputs)
                # print("x",y_pred.shape)
                # print("y",y.shape)
                preds=torch.reshape(preds,(-1,model.vocab_size))
                labels=torch.reshape(labels,(-1,))
                # print("x",y_pred.shape)
                # print("y",y.shape)
                loss = loss_fn(preds, labels)
                print(loss)
                loss.backward() # calculate gradients
                loss_tot+=loss
                total_loss+=loss*
                num_batches+=1
                batchnum+=1
                # nn.utils.clip_grad_norm_(model.parameters(), 20)
                optimizer.step() 
               
                # idx=labels!=0
                # pred=torch.argmax(preds,dim=1)
                # pred=pred[idx]
                # labels=labels[idx]
                # acc=(labels==pred).sum().item()
                # accuracy_total+=acc
                del inputs
                del labels
                del preds

                gc.collect()
            # accuracy=float(accuracy_total)/float(total_units)
            perp=float(loss_tot)/float(batchnum)
            perp=math.exp(perp)

            experiment.log_metric("per_epoch_perplexity",perp)
            

                
        avg_loss=float(total_loss)/float(num_batches)
        perplexity=math.exp(avg_loss) 
        #perplex=torch.exp(total_loss/word_count)
        print("perplexity:", perplexity)
        #print("perplex:",perplex)
        experiment.log_metric("perplexity", perplexity)
        
        
        
        
    pass
def test(experiment, model,hyper_params,test_loader,GPT):
    if not GPT:
        loss_fn=nn.CrossEntropyLoss(ignore_index=hyper_params["vocab_size"])
   
       
    
    model = model.eval()
    total_loss = 0
    word_count = 0
    num_batches=0

    with experiment.test():
        with torch.no_grad():
            accuracy_total=0
            for batch in tqdm(test_loader):
                inputs=batch["inputs"].to(device)
                labels=batch["labels"].to(device)
                lens=batch["lengths"].to(device)
                masks=batch["masks"].to(device)
                word_count+=torch.sum(lens)

                #print(x.shape)
                if GPT:
                    # out=model(inputs,masks)
                    # preds=out.logits
                    print(inputs.shape)
                    inner=inputs
                    labs=labels
                    mass=masks
                    loss = model(inner,mass).loss
                    print(loss)
                    
                    # preds=torch.reshape(preds,(-1,model.vocab_size))
                    # labels=torch.reshape(labs,(-1,))
                    
                    # idx=labels!=hyper_params["vocab_size"]
                    # pred=torch.argmax(preds,dim=1)
                    # pred=pred[idx]
                    # labels=labels[idx]
                    # acc=(labels==pred).sum().item()
                    # accuracy_total+=acc
                
                    
                    
                    # print(out.logits.shape[2])
                else:
                    inner=inputs[:,0:-1]
                    labs=labels[:,1:]
                    preds = model(inner)
                    # print("x",y_pred.shape)
                    # print("y",y.shape)

                    preds=torch.reshape(preds,(-1,model.vocab_size))
                    labels=torch.reshape(labs,(-1,))
                    
                    # idx=labels!=50257
                    # pred=torch.argmax(preds,dim=1)
                    # pred=pred[idx]
                    # labels=labels[idx]
                    # acc=(labels==pred).sum().item()
                    # accuracy_total+=acc
                # print("x",y_pred.shape)
                # print("y",y.shape)
                    loss = loss_fn(preds, labels)

                total_loss+=loss
                # total_loss+=loss*torch.sum(lens)
                num_batches+=1
                # nn.utils.clip_grad_norm_(model.parameters(), 20)

        
        # accuracy=float(accuracy_total)/float(word_count)
        avg_loss=float(total_loss)/float(num_batches)
        perplexity=math.exp(avg_loss) 
        #perplex=torch.exp(total_loss/word_count)
        print("perplexity:", perplexity)
        #print("perplex:",perplex)
        experiment.log_metric("perplexity", perplexity)
        # experiment.log_metric("accuracy",accuracy)
        pass
        # Log perplexity to Comet.ml using experiment.log_metric
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file")
    parser.add_argument("test_file")
    parser.add_argument("-m", "--model", type=str, default="",
                        help="transformer or gpt2")
    parser.add_argument("-bs", "--batch_size", type=int, default=1,
                        help="batch size")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    args = parser.parse_args()

    # Load the GPT2 Tokenizer
   
    # Load the train, test DataLoader NOTE: Parse the data using the GPT2 tokenizer for both models
    
    vocab_size=50258

    if args.model == "transformer":
        # Load your transformer
        GPT=False
        model=Transformer(vocab_size,hyper_params["embedding_size"],hyper_params["window_size"],device).to(device)
        
    elif args.model == "gpt2":
        GPT=True
        # Load the GPT2 model
        
        model=GPT2_Transformer().to(device)
        pass
    tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
    train_file="penn-UNK-train.txt"
    test_file="penn-UNK-test.txt"
    train_loader,test_loader=load_dataset(train_file,test_file,hyper_params["window_size"],tokenizer,hyper_params["batch_size"],GPT)
    
    torch.cuda.empty_cache()
    if not GPT:
        train(experiment, model,hyper_params,train_loader)
    # Train the model if args.model == "transformer"
    test(experiment,model,hyper_params,test_loader,GPT)

    # Test the model on the test set - report perplexity
    

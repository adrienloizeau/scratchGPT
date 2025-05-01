import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

torch.manual_seed(123)

class CharTokenizer(nn.Module):
    def __init__(self, dataset):
        super(CharTokenizer, self).__init__()
        self.chars = sorted(list(set(dataset)))
        self.encode_dict = { c:i for i,c in enumerate(self.chars)}
        self.decode_dict = { i:c for i,c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        return torch.tensor([self.encode_dict[c] for c in text], dtype=torch.long)

    def decode(self, indices):
        return ''.join([self.decode_dict[i.item()] for i in indices])


class Dataloader(nn.Module):
    def __init__(self, data, batch_size=4, block_size=8, tokenizer=CharTokenizer):
        super(Dataloader, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.data = self.tokenize(data)
        self.batch_size = batch_size
        self.block_size = block_size

    def get_batch(self):
        # Generate a random batch of data
        idx = torch.randint(0, len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([self.data[i: i+ self.block_size] for i in idx])
        y = torch.stack([self.data[i+1: i + self.block_size + 1] for i in idx])
        return x, y

    def tokenize(self, text):
        # Encode the input data
        return self.tokenizer.encode(text)
        
    def detokenize(self, indices):
        # Decode the output data
        return self.tokenizer.decode(indices)


def merge_dataset(ds):
    # Merge the dataset into a single string
    split = ["train", "test", "validation"]
    full_ds = ""
    for s in split:  
        for i in range(len(ds[s]["title"])):  
            title = ds[s]["title"][i]
            body = ds[s]["body"][i]
            full_ds += title + "\n" + body + "\n\n" 
    return full_ds

def clean_dataset(ds):
    # Clean the dataset by removing unwanted characters
    unecessary_chars  = ["©","®","₹","™","–—","⇒", "…"]
    cleaned_dataset  = ds
    for u in unecessary_chars:
        cleaned_dataset = cleaned_dataset.replace(u,"")
    return cleaned_dataset


class Model(nn.Module):
    def __init__(self, vocab_size, block_size= 8, n_emb=32):
        super(Model,self).__init__()
        # table of vocab_size x vocab_size to predict the next token only based on the current token
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)         
        # self.positionnal_embedding_table = nn.Embedding(vocab_size,vocab_size)
        # self.embedding_out = nn.Embedding(n_emb, block_size)
        # self.linear = nn.Linear(vocab_size, vocab_size)
     
    def forward(self, idx, target = None):
        # For each idx in the batch we get the corresponding token
        # and compare it to the target
        # We theregore have the idx information and the positionnal information
        # idx (B, T) 
        # target (B, T) 

        logits = self.token_embedding_table(idx) # =>(B, T, C)
        # pos = self.positionnal_embedding_table(idx)
        # logits = logits + pos
        # logits = logits.view(-1, logits.shape[2])
        # logits = self.linear(logits)
        if target is not None: 
            B, T, C = logits.shape
            logits = logits.view(B * T, C) 
            loss = F.cross_entropy(logits, target.view(-1)) # Negative Log Likelihood Loss on all possible classes
            return logits, loss
        else:
            loss = None
            return logits, loss


    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            # Focus only on the last time step -> so has all the block_size
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probs
            probs = F.softmax(logits, dim = 1) # (B, C)

            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
        return idx
        
# Hyperparameters
learning_rate = 1e-3
block_size = 8
batch_size = 16 

# Load the dataset, clean and split it
ds = load_dataset("jonathanli/law-stack-exchange")
ds = merge_dataset(ds)
ds = clean_dataset(ds)
split_size = int(0.8 * len(ds))
train_data = ds[:split_size]
val_data = ds[split_size:]

# Initialize the Tokenizer and DataLoader
tokenizer = CharTokenizer(ds)
train_loader = Dataloader(train_data, batch_size=batch_size, block_size=block_size, tokenizer=tokenizer)
val_loader = Dataloader(val_data, batch_size=batch_size, block_size=block_size, tokenizer=tokenizer)
x, y = train_loader.get_batch()

# initialize the model and optimizer
m = Model(tokenizer.vocab_size, block_size=block_size)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for step in tqdm(range(10000)):
    logits, loss = m(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())
idx = torch.zeros((1,1), dtype = torch.long)
print(tokenizer.decode(m.generate(idx, max_new_tokens = 100)[0]))
# print("x shape: ", x.shape)
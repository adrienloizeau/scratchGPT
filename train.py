import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from tqdm import tqdm



# Hyperparameters
learning_rate = 1e-2
block_size = 8 # Maximum context length
batch_size = 16 # nb of indepedant blocks fead to the model in parallel
max_iters = 10000
eval_interval = 300
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200

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
    
@torch.no_grad()
def estimate_loss(split):
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Load the dataset, clean and split it
ds = load_dataset("jonathanli/law-stack-exchange")
ds = merge_dataset(ds)
ds = clean_dataset(ds)
split_size = int(0.8 * len(ds))
train_data = ds[:split_size]
val_data = ds[split_size:]

# Initialize the Tokenizer and DataLoader
tokenizer = CharTokenizer(ds)

train_data = tokenizer.encode(train_data)
val_data = tokenizer.encode(val_data)
train_data = torch.tensor(train_data, dtype=torch.long)
val_data = torch.tensor(val_data, dtype=torch.long)

# initialize the model and optimizer
m = Model(tokenizer.vocab_size, block_size=block_size)
m = m.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
for iter in tqdm(range(max_iters)):
   # evaluate the loss on train/val sets and write checkpoints
    if iter % eval_interval == 0 :
        losses = estimate_loss("val")
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # train_loader = Dataloader(train_data, batch_size=batch_size, block_size=block_size, tokenizer=tokenizer)
    x, y = get_batch('train')
    logits, loss = m(x,y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Loss: ", loss.item())

print("Training finished, generating text...")
context = torch.zeros((1,1), dtype = torch.long).to(device)
print(tokenizer.decode(m.generate(context, max_new_tokens = 100)[0]))

# print("x shape: ", x.shape)
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from configs.config import BaseConfig, LargeConfig

# Set the random seed for reproducibility
torch.manual_seed(123)

config = BaseConfig()  # or LargeConfig() for larger model
# Hyperparameters
learning_rate = config.learning_rate
block_size = config.block_size  # Maximum context length
batch_size = config.batch_size  # Number of independent blocks fed to the model in parallel
max_iters = config.max_iters
eval_interval = config.eval_interval
device = config.device
eval_iters = config.eval_iters
save_path = config.save_path
mode = config.mode
n_embd = config.n_embd
nb_heads = config.nb_heads
nb_layers = config.nb_layers

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


class Head(nn.Module):
    def __init__(self, head_size):
        super(Head, self).__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # keeps information about the token
        self.query = nn.Linear(n_embd, head_size, bias=False) # keeps information about the position
        self.value = nn.Linear(n_embd, head_size, bias=False) # keeps information about the value of each token to the position
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.2)
        self.head_size = head_size
        self.scale = head_size ** -0.5

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        v = self.value(x) # (B, T, C)
        # compute attention scores 
        wei = (q @ k.transpose(-2, -1)) * self.scale # scale for numerical stability
        # mask out the future positions
        mask = self.tril[:T, :T].unsqueeze(0) # create a mask of shape (1, T, T)
        wei = wei.masked_fill(mask == 0, float("-inf")) # masks the future positions
        # apply softmax to get the attention weights
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.dropout = nn.Dropout(0.2)
        self.proj = nn.Linear(n_embd, n_embd)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(head_size, n_heads)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) 
        x = x + self.ffwd(self.ln2(x))
        return x
    
class Model(nn.Module):
    def __init__(self, vocab_size, block_size= 8, n_embd=32, nb_heads=4, nb_layers=4):
        super(Model,self).__init__()
        # table of vocab_size x vocab_size to predict the next token only based on the current token
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)         
        self.positionnal_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_blocks = nn.Sequential(*[Block(n_embd, nb_heads) for _ in range(nb_layers)]) # 4 blocks of self attention
        self.ln = nn.LayerNorm(n_embd)
        self.feed_forward = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # self.embedding_out = nn.Embedding(n_emb, block_size)
     
    def forward(self, idx, target = None):
        # For each idx in the batch we get the corresponding token
        # and compare it to the target
        # We theregore have the idx information and the positionnal information
        # idx (B, T) 
        # target (B, T)

        B, T = idx.shape
        token_embedding = self.token_embedding_table(idx) # (B, T, C)
        pos_embedding = self.positionnal_embedding_table(torch.arange(T, device = device)) # (T, C)
        x = token_embedding + pos_embedding # (B, T, C)
        x=  self.sa_blocks(x)
        x = self.ln(x) # (B, T, C)
        x = self.feed_forward(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
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

            idx_cond = idx[:, -block_size:] # (B, T) -> (B, block_size)
            logits, loss = self(idx_cond)
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


if __name__ == "__main__":
    # Load the dataset, clean and split it
    dataset_path = "data/cleaned_documents.jsonl"
    ds = load_dataset("json", data_files=dataset_path)["train"]["text"]
    ds = ''.join(ds)
    split_size = int(0.8 * len(ds))
    train_data, val_data = ds[:split_size], ds[split_size:]

    # Initialize the Tokenizer and DataLoader
    tokenizer = CharTokenizer(ds)

    train_data = tokenizer.encode(train_data)
    val_data = tokenizer.encode(val_data)
    train_data = torch.tensor(train_data, dtype=torch.long)
    val_data = torch.tensor(val_data, dtype=torch.long)

    # initialize the model and optimizer
    if mode == "scratch":
        m = Model(tokenizer.vocab_size, block_size=block_size, n_embd=n_embd, nb_heads=nb_heads, nb_layers=nb_layers)
        m = m.to(device)
    elif mode == "pretrained":
        m = Model(tokenizer.vocab_size, block_size=block_size, n_embd=n_embd, nb_heads=nb_heads, nb_layers=nb_layers)
        m = torch.load(save_path, map_location=device)
        
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    best_val = float('inf')
    for iter in tqdm(range(max_iters)):
    # evaluate the loss on train/val sets and write checkpoints
        if iter % eval_interval == 0 :
            losses = estimate_loss("val")
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_val:
                best_val = losses['val']
                torch.save(m, save_path)

        x, y = get_batch('train')
        logits, loss = m(x,y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()    

    print("Loss: ", loss.item())

    print("Training finished, generating text...")
    custom_text = input("Enter custom text to start generation (leave empty for random): ")
    if custom_text:
        context = tokenizer.encode(custom_text).unsqueeze(0).to(device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long).to(device)
    print(tokenizer.decode(m.generate(context, max_new_tokens=500)[0]))

    # print("x shape: ", x.shape)
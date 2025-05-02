import torch 
from torch import nn
import torch.nn.functional as F

from configs.config import BaseConfig, LargeConfig

config = LargeConfig()  # or BaseConfig() for smaller model
device = config.device
block_size = config.block_size  # Maximum context length
batch_size = config.batch_size  # Number of independent blocks fed to the model in parallel
n_embd = config.n_embd
nb_heads = config.nb_heads
nb_layers = config.nb_layers

class CharTokenizer(nn.Module):
    def __init__(self, dataset=None, load_path=None):
        super(CharTokenizer, self).__init__()
        assert dataset is not None or load_path is not None, "Either dataset or load_path must be provided"
        if dataset is not None:
            self.chars = sorted(list(set(dataset + '\n')))
        else:
            self.load(load_path)
        self.encode_dict = {c: i for i, c in enumerate(self.chars)}
        self.decode_dict = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        # Utiliser <UNK> pour les caractères inconnus
        return torch.tensor([self.encode_dict.get(c, self.encode_dict[" "]) for c in text], dtype=torch.long)

    def decode(self, indices):
        return ''.join([self.decode_dict[i.item()] for i in indices])
    
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            for char in self.chars:
                f.write(f"{repr(char)}\n")

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.chars = [eval(line.strip()) for line in f]
        # Ajouter <UNK> si absent
        if '<UNK>' not in self.chars:
            self.chars.append('<UNK>')
        self.encode_dict = {c: i for i, c in enumerate(self.chars)}
        self.decode_dict = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)


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
    
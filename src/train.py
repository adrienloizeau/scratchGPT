import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from src.config import BaseConfig, LargeConfig
from model import Model, CharTokenizer, Block, MultiHeadAttention, FeedForward, Head
import os

# Set the random seed for reproducibility
torch.manual_seed(123)

config = LargeConfig()  # or LargeConfig() for larger model
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
    import wandb
    wandb.init(
        project="finetune-instruct",
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "block_size": block_size,
            "n_embd": n_embd,
            "nb_heads": nb_heads,
            "nb_layers": nb_layers,
        }
    )
    # Load the dataset, clean and split it
    dataset_path = "data/cleaned_dataset.jsonl"
    ds = load_dataset("json", data_files=dataset_path)["train"]["text"]
    ds = ''.join(ds)
    split_size = int(0.8 * len(ds)) 
    train_data, val_data = ds[:split_size], ds[split_size:]

    # Initialize the Tokenizer and DataLoader
    tokenizer = CharTokenizer(ds)
    tokenizer.save("tokenizer_training.txt")

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
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                print(f"Saving model to {save_path}")
                torch.save(m, save_path)

        x, y = get_batch('train')
        logits, loss = m(x,y)
        wandb.log({"train_loss": loss.item(), "iteration": iter, "val_loss": losses['val']})
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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from scratchgpt.models.model import Model, CharTokenizer
import os
import wandb
import hydra
from omegaconf import DictConfig
CONFIG_PATH = "pkg://scratchgpt.configs.hydra"

@torch.no_grad()
def estimate_loss(split, m, eval_iters, device='cpu', block_size=128, batch_size=64, data=None):
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(block_size=block_size, batch_size=batch_size, data=data, device=device)
            _, loss = m(X, Y, device)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

def get_batch(block_size=128, batch_size=64, data=None, device='cpu'):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def train(cfg: DictConfig):

    # Set the random seed for reproducibility
    torch.manual_seed(cfg.train.seed)

    # Hyperparameters
    learning_rate = cfg.model.learning_rate
    block_size = cfg.model.block_size  # Maximum context length
    batch_size = cfg.model.batch_size  # Number of independent blocks fed to the model in parallel
    max_iters = cfg.model.max_iters
    eval_interval = cfg.model.eval_interval
    device = cfg.model.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to cpu.")
        device = "cpu"
    eval_iters = cfg.model.eval_iters
    save_path = cfg.model.save_path
    mode = cfg.model.mode
    n_embd = cfg.model.n_embd
    nb_heads = cfg.model.nb_heads
    nb_layers = cfg.model.nb_layers

    if cfg.train.wandb.enabled:
        wandb.init(
            project=cfg.train.wandb.project,
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
    dataset_path = cfg.data.dataset_path
    ds = load_dataset("json", data_files=dataset_path)["train"]["text"]
    ds = ''.join(ds)
    split_size = int(cfg.data.train_split * len(ds))
    train_data, val_data = ds[:split_size], ds[split_size:]

    # Initialize the Tokenizer and DataLoader
    tokenizer = CharTokenizer(ds)
    tokenizer.save(cfg.data.tokenizer_path)

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
        
    optimizer = torch.optim.AdamW(m.parameters(), lr=cfg.train.optimizer.lr)
    best_val = float('inf')
    for iter in tqdm(range(max_iters)):
        if iter % eval_interval == 0 :
            losses = estimate_loss("val", m, eval_iters, device, block_size, batch_size, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_val:
                best_val = losses['val']
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                print(f"Saving model to {save_path}")
                torch.save(m, save_path)

        x, y = get_batch(block_size=block_size, batch_size=batch_size, data=train_data, device=device)
        logits, loss = m(x,y)
        if cfg.train.wandb.enabled:
            wandb.log({"train_loss": loss.item(), "iteration": iter, "val_loss": losses['val']})
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()    

    print("Loss: ", loss.item())

    print("Training finished, generating text...")
    if cfg.train.prompt_on_finish:
        custom_text = input("Enter custom text to start generation (leave empty for random): ")
        if custom_text:
            context = tokenizer.encode(custom_text).unsqueeze(0).to(device)
        else:
            context = torch.zeros((1, 1), dtype=torch.long).to(device)
        print(tokenizer.decode(m.generate(context, max_new_tokens=cfg.train.max_new_tokens, block_size=block_size)[0]))

if __name__ == "__main__":
    train()

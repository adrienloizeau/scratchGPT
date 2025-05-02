from train import  Model, Block, MultiHeadAttention, FeedForward, Head
from train import CharTokenizer
from datasets import load_dataset
import torch
import torch.nn as nn


# Hyperparameters
learning_rate = 1e-2
block_size = 8 # Maximum context length
batch_size = 16 # nb of indepedant blocks fead to the model in parallel
max_iters = 10000
eval_interval = 300
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32
nb_heads = 4
nb_layers = 4
save_path = "model.pt"

# Load the dataset, clean and split it
dataset_path = "data/cleaned_documents.jsonl"
ds = load_dataset("json", data_files=dataset_path)["train"]["text"]

# Initialize the Tokenizer and DataLoader
tokenizer = CharTokenizer(ds)

m = Model(tokenizer.vocab_size, block_size=block_size, n_embd=n_embd, nb_heads=nb_heads, nb_layers=nb_layers)
m = torch.load(save_path, map_location=device)
context = torch.zeros((1,1), dtype = torch.long).to(device)
print(tokenizer.decode(m.generate(context, max_new_tokens = 500)[0]))
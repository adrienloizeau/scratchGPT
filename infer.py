from train import  Model, Block, MultiHeadAttention, FeedForward, Head
from train import CharTokenizer
from datasets import load_dataset
import torch
import torch.nn as nn
from configs.config import BaseConfig, LargeConfig


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

# Load the dataset, clean and split it
dataset_path = "data/cleaned_dataset.jsonl"
ds = load_dataset("json", data_files=dataset_path)["train"]["text"]
ds = ''.join(ds)
# Initialize the Tokenizer and DataLoader

tokenizer = CharTokenizer(ds)

m = Model(tokenizer.vocab_size, block_size=block_size, n_embd=n_embd, nb_heads=nb_heads, nb_layers=nb_layers)
m = torch.load(save_path, map_location=device, weights_only=False)
context = torch.zeros((1,1), dtype = torch.long).to(device)
print("Generated Text:")
print(tokenizer.decode(m.generate(context, max_new_tokens = 500)[0]))

# Question Answering
question = "What is the purpose of this model?"
question_tokens = tokenizer.encode(question)
context_with_question = torch.tensor(question_tokens, dtype=torch.long).unsqueeze(0).to(device)
print("\nAnswer to the Question:")
print(tokenizer.decode(m.generate(context_with_question, max_new_tokens=100)[0]))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from train import Model, Block, MultiHeadAttention, FeedForward, Head, CharTokenizer
from datasets import load_dataset
from tqdm import tqdm
from configs.config import BaseConfig, LargeConfig  # Assuming you have a config.py with these classes

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
num_epochs = 10  # Number of epochs for training

torch.manual_seed(123)

# Dataset class for instruction-based training
class InstructionDataset(Dataset):
    def __init__(self, prompts, responses, tokenizer, block_size):
        self.data = [
            tokenizer.encode(f"{prompt} <|endoftext|> {response}")
            for prompt, response in zip(prompts, responses)
        ]
        self.block_size = block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        # Truncate or pad the sequence to block_size
        if len(x) > self.block_size:
            x = x[:self.block_size]
        else:
            padding = torch.zeros(self.block_size - len(x), dtype=torch.long)  # Create padding as a tensor
            x = torch.cat((torch.tensor(x, dtype=torch.long), padding))  # Concatenate x with padding
        # Shift for the target
        y = torch.cat((x[1:], torch.tensor([0], dtype=torch.long)))  # Shifted version of x for the target
        return x, y

# Function to evaluate the model
@torch.no_grad()
def evaluate(model, tokenizer, prompts, responses):
    model.eval()
    for prompt, true_response in zip(prompts[:5], responses[:5]):  # Evaluate on a few examples
        input_ids = tokenizer.encode(prompt).unsqueeze(0).to(device)
        generated_ids = model.generate(input_ids, max_new_tokens=100)
        generated_response = tokenizer.decode(generated_ids[0])
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_response}")
        print(f"True: {true_response}")
        print("-" * 50)


if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset("databricks/databricks-dolly-15k")

    # Extract prompts and responses
    prompts = dataset["train"]["instruction"]
    responses = dataset["train"]["response"]

    # Initialize the Tokenizer
    tokenizer = CharTokenizer("".join(prompts + responses) + "<|prompt|><|response|>")

    # Prepare the dataset
    train_dataset = InstructionDataset(prompts, responses, tokenizer, block_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    m = Model(tokenizer.vocab_size, block_size=block_size, n_embd=n_embd, nb_heads=nb_heads, nb_layers=nb_layers)
    m = torch.load(save_path, map_location=device)
    model = m.to(device)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids, targets = batch
            input_ids, targets = input_ids.to(device), targets.to(device)

            logits, loss = model(input_ids, targets)
             # Focus the loss on the response portion
            response_start = (input_ids == tokenizer.encode("<|response|>")[0]).nonzero(as_tuple=True)[1][0]
            loss = nn.functional.cross_entropy(
                logits[:, response_start:].reshape(-1, logits.size(-1)),
                targets[:, response_start:].reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        # Save the model if validation loss improves
        if loss.item() < best_val_loss:
            best_val_loss = loss.item()
            torch.save(model.state_dict(), save_path)

    print("Training finished!")

    # Evaluate the model
    print("Evaluating the model...")
    evaluate(model, tokenizer, prompts, responses)

    # Generate text
    print("Generating text...")
    custom_text = input("Enter custom text to start generation (leave empty for random): ")
    if custom_text:
        context = tokenizer.encode(custom_text).unsqueeze(0).to(device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long).to(device)
    generated_ids = model.generate(context, max_new_tokens=100)
    print(tokenizer.decode(generated_ids[0]))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import Model, CharTokenizer, Block, MultiHeadAttention, FeedForward, Head
from datasets import load_dataset
from tqdm import tqdm
from configs.config import LargeConfig

# Configuration
config = LargeConfig()
learning_rate = config.learning_rate
block_size = config.block_size
batch_size = config.batch_size
device = config.device
save_path = "weights/instruct.pt"
n_embd = config.n_embd
nb_heads = config.nb_heads
nb_layers = config.nb_layers
num_epochs = 3

torch.manual_seed(123)

# Dataset class for instruction-based training
class InstructionDataset(Dataset):
    def __init__(self, prompts, responses, tokenizer, block_size):
        self.data = [
            tokenizer.encode(f"|||{prompt}|||{response}|||")
            for prompt, response in zip(prompts, responses)
        ]
        self.block_size = block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if len(x) > self.block_size:
            x = x[:self.block_size]
        else:
            padding = torch.zeros(self.block_size - len(x), dtype=torch.long)
            x = torch.cat((torch.as_tensor(x, dtype=torch.long), padding))
        y = torch.cat((x[1:], torch.tensor([0], dtype=torch.long)))
        return x, y

# Function to evaluate the model
@torch.no_grad()
def evaluate(model, tokenizer, prompts, responses):
    model.eval()
    for prompt, true_response in zip(prompts[:5], responses[:5]):
        input_ids = tokenizer.encode(prompt).unsqueeze(0).to(device)
        generated_ids = model.generate(input_ids, max_new_tokens=100)
        generated_response = tokenizer.decode(generated_ids[0])
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_response}")
        print(f"True: {true_response}")
        print("-" * 50)


if __name__ == "__main__":
    import wandb
    # Initialize wandb
    wandb.init(
        project="finetune-instruct",
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "block_size": block_size,
            "num_epochs": num_epochs,
            "n_embd": n_embd,
            "nb_heads": nb_heads,
            "nb_layers": nb_layers,
        }
    )

    # Load the dataset
    dataset = load_dataset("databricks/databricks-dolly-15k")
    prompts = dataset["train"]["instruction"]
    responses = dataset["train"]["response"]

    # Initialize the tokenizer
    tokenizer = CharTokenizer(load_path="tokenizer_training.txt")
    print(tokenizer.vocab_size)

    # Prepare the dataset and dataloader
    train_dataset = InstructionDataset(prompts, responses, tokenizer, block_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = torch.load("weights/model.pt", map_location=device, weights_only=False)
    model = model.to(device)

    # Initialize the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids, targets = batch
            input_ids, targets = input_ids.to(device), targets.to(device)

            # Forward pass
            logits = model(input_ids)
            if isinstance(logits, tuple):
                logits = logits[0]

            # Calcul de la perte
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        # Log metrics to wandb
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})

        # Save the model if validation loss improves
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            wandb.log({"best_loss": best_val_loss})

    print("Training finished!")
    wandb.finish()
    
    # Evaluate the model
    print("Generating text...")
    custom_text = input("Enter custom text to start generation (leave empty for random): ").strip()
    if custom_text:
        custom_text = f"|||{custom_text}|||"
        context = tokenizer.encode(custom_text).unsqueeze(0).to(device)
    else:
        default_prompt = "|||What is the purpose of life?|||"
        print(f"No input provided. Using default prompt: {default_prompt}")
        context = tokenizer.encode(default_prompt).unsqueeze(0).to(device)

    generated_ids = model.generate(context, max_new_tokens=100)
    print(tokenizer.decode(generated_ids[0]))   
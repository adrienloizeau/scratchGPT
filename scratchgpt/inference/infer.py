from scratchgpt.models.model import Model, CharTokenizer
from datasets import load_dataset
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
CONFIG_PATH = "pkg://scratchgpt.configs.hydra"

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def infer(cfg: DictConfig):

    # Hyperparameters
    block_size = cfg.model.block_size  # Maximum context length
    device = cfg.model.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to cpu.")
        device = "cpu"
    save_path = cfg.model.save_path
    n_embd = cfg.model.n_embd
    nb_heads = cfg.model.nb_heads
    nb_layers = cfg.model.nb_layers

    # Load the dataset, clean and split it
    dataset_path = cfg.data.dataset_path
    ds = load_dataset("json", data_files=dataset_path)["train"]["text"]
    ds = ''.join(ds)
    # Initialize the Tokenizer and DataLoader

    tokenizer = CharTokenizer(ds)

    m = Model(tokenizer.vocab_size, block_size=block_size, n_embd=n_embd, nb_heads=nb_heads, nb_layers=nb_layers)
    m = torch.load(save_path, map_location=device, weights_only=False)
    context = torch.zeros((1,1), dtype = torch.long).to(device)
    print("Generated Text:")
    print(tokenizer.decode(m.generate(context, max_new_tokens=cfg.train.max_new_tokens, block_size=block_size)[0]))

    # Question Answering
    question = "What is the purpose of this model?"
    question_tokens = tokenizer.encode(question)
    context_with_question = torch.tensor(question_tokens, dtype=torch.long).unsqueeze(0).to(device)
    print("\nAnswer to the Question:")
    print(tokenizer.decode(m.generate(context_with_question, max_new_tokens=100, block_size=block_size)[0]))


if __name__ == "__main__":
    # Set the random seed for reproducibility
    infer()

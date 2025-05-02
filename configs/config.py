import torch

class BaseConfig:
    # Hyperparameters
    learning_rate = 1e-2
    block_size = 8 #    Maximum context length
    batch_size = 16 # nb of indepedant blocks fead to the model in parallel
    max_iters = 5000
    eval_interval = 300
    device = "cpu"
    eval_iters = 200
    n_embd = 32
    nb_heads = 4
    nb_layers = 4
    save_path = "model.pt"
    mode = "scratch"

class LargeConfig:
    # Hyperparameters
    learning_rate = 3e-4
    block_size = 256 # Maximum context length
    batch_size = 64 # nb of indepedant blocks fead o the model in parallel
    max_iters = 10000
    eval_interval = 300
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_iters = 200
    save_path = "weights/model.pt"
    mode = "scratch"
    n_embd = 384
    nb_heads = 6
    nb_layers = 6

import gradio as gr
from model import Model, CharTokenizer, Block, MultiHeadAttention, FeedForward, Head
import torch
from scratchgpt.configs.config import LargeConfig

# Load the model
config = LargeConfig()
learning_rate = config.learning_rate
block_size = config.block_size
batch_size = config.batch_size
device = config.device
save_path = "weights/instruct.pt"
n_embd = config.n_embd
nb_heads = config.nb_heads
nb_layers = config.nb_layers


device = config.device
model_path = "weights/instruct.pt"

# Initialize the tokenizer
tokenizer = CharTokenizer(load_path="tokenizer_training.txt")

# Initialize the model
model = torch.load("weights/model.pt", map_location=device, weights_only=False)
model = model.to(device)

def response(prompt, history):
    # Encode the prompt
    input_ids = tokenizer.encode(prompt).unsqueeze(0).to(device)
    
    # Generate a response
    generated_ids = model.generate(input_ids, max_new_tokens=100)
    
    # Decode the generated response
    generated_response = tokenizer.decode(generated_ids[0])

    
    # Return the updated history
    return {"role": "assistant", "content": generated_response}

gr.ChatInterface(
    fn=response, 
    type="messages"
).launch()
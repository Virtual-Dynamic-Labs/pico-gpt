import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import time
import pickle
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.pico_gpt import GPT, GPTConfig
from src.tokenizer import SimpleTokenizer


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, data, eval_iters, batch_size, block_size, device):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, batch_size, block_size, device)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()


def train():
    # Configuration
    batch_size = 16
    block_size = 256
    max_iters = 5000
    eval_interval = 1000
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    
    # Model configuration
    config = GPTConfig()
    config.block_size = block_size
    config.vocab_size = 50304
    config.n_layer = 6
    config.n_head = 6
    config.n_embd = 384
    config.dropout = 0.2
    
    print(f"Using device: {device}")
    
    # Load or create sample data
    try:
        with open('shakespeare.txt', 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        # Create sample text if no data file
        text = """Hello world! This is a sample text for training our pico GPT model.
        The model will learn to predict the next character given the previous characters.
        This is a very basic example of how language models work.
        In practice, you would use much larger datasets like books, articles, or web text.
        """ * 100  # Repeat to have enough data
    
    # Tokenize the data
    tokenizer = SimpleTokenizer()
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    # Split into train and validation
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Dataset size: {len(data):,} tokens")
    print(f"Train size: {len(train_data):,} tokens")
    print(f"Val size: {len(val_data):,} tokens")
    
    # Initialize model
    model = GPT(config)
    model.to(device)
    print(f"Model parameters: {model.get_num_params()/1e6:.2f}M")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    for iter in range(max_iters):
        # Evaluate the loss on train/val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            train_loss = estimate_loss(model, train_data, eval_iters, batch_size, block_size, device)
            val_loss = estimate_loss(model, val_data, eval_iters, batch_size, block_size, device)
            print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
        
        # Sample a batch of data
        xb, yb = get_batch(train_data, batch_size, block_size, device)
        
        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer': tokenizer
    }, 'pico_gpt_model.pt')
    
    print("Training complete! Model saved as 'pico_gpt_model.pt'")
    
    # Generate some text
    model.eval()
    context = torch.tensor(tokenizer.encode("Hello"), dtype=torch.long, device=device).unsqueeze(0)
    generated = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=10)
    generated_text = tokenizer.decode(generated[0].tolist())
    print(f"\nGenerated text: {generated_text}")


if __name__ == "__main__":
    train()
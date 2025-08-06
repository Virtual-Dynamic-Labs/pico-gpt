#!/usr/bin/env python3
"""
Train a smart reasoning model with balanced parameters
Optimized for intelligence while keeping reasonable training time
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import time
import pickle
import os
import math
import sys

# Add parent directory to path for imports
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
def estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size, device):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    # Linear warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # Decay
    if it > lr_decay_iters:
        return min_lr
    # Cosine decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def train_smart_model():
    """Train a smart reasoning model"""
    print("*** Training Smart Reasoning Model ***")
    print("=" * 45)
    
    # Balanced training hyperparameters
    batch_size = 16
    block_size = 256
    max_iters = 1500    # Reasonable number of iterations
    eval_interval = 300
    learning_rate = 3e-4
    warmup_iters = 100
    lr_decay_iters = 1500
    min_lr = 3e-5
    eval_iters = 100
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Smart model configuration - balanced for intelligence
    config = GPTConfig()
    config.block_size = block_size
    config.vocab_size = 3000
    config.n_layer = 10      # Good depth for reasoning
    config.n_head = 10       # Good attention
    config.n_embd = 640      # Balanced embedding size
    config.dropout = 0.1
    config.bias = True
    
    print(f"Model configuration:")
    print(f"  - Layers: {config.n_layer}")
    print(f"  - Heads: {config.n_head}")
    print(f"  - Embedding dim: {config.n_embd}")
    print(f"  - Context length: {config.block_size}")
    print(f"  - Vocabulary size: {config.vocab_size}")
    
    # Load training data
    data_path = os.path.join('..', 'datasets', 'smart_reasoning_data.txt')
    print(f"\nLoading training data from: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Data size: {len(text):,} characters")
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    
    # Encode the data
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Encoded data: {len(data):,} tokens")
    
    # Split into train and validation
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Training tokens: {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")
    
    # Initialize model
    print(f"\nInitializing model...")
    model = GPT(config)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    print(f"\nStarting training for {max_iters:,} iterations...")
    print("-" * 45)
    
    # Training variables
    best_val_loss = float('inf')
    start_time = time.time()
    
    # Training loop
    for iter_num in range(max_iters):
        # Update learning rate
        lr = get_lr(iter_num, warmup_iters, learning_rate, lr_decay_iters, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate periodically
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size, device)
            elapsed_time = time.time() - start_time
            
            print(f"iter {iter_num:4d}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}, lr {lr:.2e}, "
                  f"time {elapsed_time:.1f}s")
            
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'tokenizer': tokenizer,
                    'iter': iter_num,
                    'best_val_loss': best_val_loss,
                    'train_loss': losses['train']
                }
                torch.save(checkpoint, '../models/pico_gpt_large.pt')
                print(f"  -> Saved new best model (val loss: {best_val_loss:.4f})")
        
        # Training step
        xb, yb = get_batch(train_data, batch_size, block_size, device)
        
        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
    
    # Final save
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer': tokenizer,
        'iter': max_iters,
        'best_val_loss': best_val_loss,
        'train_loss': loss.item()
    }
    torch.save(final_checkpoint, '../models/pico_gpt_large.pt')
    
    total_time = time.time() - start_time
    print(f"\n*** Training completed! ***")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved as: ../models/pico_gpt_large.pt")
    
    # Test generation
    print(f"\n*** Testing generation... ***")
    model.eval()
    
    test_prompts = [
        "Problem: Calculate the area of a triangle with sides 3, 4, and 5.",
        "Problem: A train travels 240 km in 3 hours. What is its speed?",
        "Problem: Debug this Python code that has an error.",
        "Problem: Solve this step by step: 2x + 5 = 3x - 7"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            generated = model.generate(
                context, 
                max_new_tokens=100,
                temperature=0.8,
                top_k=50
            )
        
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"Generated: {generated_text[len(prompt):]}")


if __name__ == "__main__":
    train_smart_model()
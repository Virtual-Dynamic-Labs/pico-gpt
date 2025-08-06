#!/usr/bin/env python3
"""
Fast test training for the smart reasoning model
Reduced parameters for quick validation
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


def train_fast_test():
    """Fast test training with reduced parameters"""
    print("*** Fast Test Training - Smart Reasoning Model ***")
    print("=" * 55)
    
    # Fast training hyperparameters
    batch_size = 8
    block_size = 128  # Shorter context for speed
    max_iters = 500   # Much fewer iterations for testing
    eval_interval = 100
    learning_rate = 3e-4
    eval_iters = 50   # Fewer eval iterations
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Smaller model configuration for fast testing
    config = GPTConfig()
    config.block_size = block_size
    config.vocab_size = 2000  # Smaller vocab for speed
    config.n_layer = 6        # Fewer layers for speed
    config.n_head = 6         # Fewer heads
    config.n_embd = 384       # Smaller embedding
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
    
    # Use only first part for fast testing
    if len(text) > 50000:
        text = text[:50000]
        print(f"Using first 50K characters for fast testing")
    
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print(f"\nStarting fast test training for {max_iters:,} iterations...")
    print("-" * 50)
    
    # Training variables
    best_val_loss = float('inf')
    start_time = time.time()
    
    # Training loop
    for iter_num in range(max_iters):
        # Evaluate periodically
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size, device)
            elapsed_time = time.time() - start_time
            
            print(f"iter {iter_num:4d}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}, time {elapsed_time:.1f}s")
            
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
        
        # Gradient clipping for stability
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
    print(f"\n*** Fast test training completed! ***")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved as: ../models/pico_gpt_large.pt")
    
    # Test generation
    print(f"\n*** Testing generation... ***")
    model.eval()
    
    test_prompts = [
        "Problem: Calculate the area of a triangle with sides 3, 4, and 5.",
        "Problem: A train travels 240 km in 3 hours. What is its speed?",
    ]
    
    for prompt in test_prompts[:2]:  # Only test 2 prompts for speed
        print(f"\nPrompt: {prompt}")
        context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            generated = model.generate(
                context, 
                max_new_tokens=50,  # Shorter generation for speed
                temperature=0.8,
                top_k=40
            )
        
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"Generated: {generated_text[len(prompt):]}")


if __name__ == "__main__":
    train_fast_test()
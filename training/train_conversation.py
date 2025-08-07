#!/usr/bin/env python3
"""
Optimized training script for conversational AI
Focused on natural dialogue and conversation patterns
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


def train_conversation_model():
    """Train an optimized conversational model"""
    print("*** Training Conversational AI Model ***")
    print("=" * 45)
    
    # Fixed hyperparameters for single response model
    batch_size = 6        # Smaller batch for larger context
    block_size = 256      # Larger context for complete responses
    max_iters = 2000      # Adequate training
    eval_interval = 100   # Regular evaluation
    learning_rate = 2e-4  # Slightly lower for stability
    warmup_iters = 200    # Proper warmup
    lr_decay_iters = 1800
    min_lr = 2e-5
    eval_iters = 20
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Optimized model for single complete responses
    config = GPTConfig()
    config.block_size = block_size
    config.vocab_size = 1500      # Good vocab size
    config.n_layer = 8            # More layers for better single responses
    config.n_head = 8             # Matching heads
    config.n_embd = 512           # Larger embeddings for better quality
    config.dropout = 0.1          # Good regularization
    config.bias = True
    
    print(f"Conversation model configuration:")
    print(f"  - Layers: {config.n_layer}")
    print(f"  - Heads: {config.n_head}")
    print(f"  - Embedding dim: {config.n_embd}")
    print(f"  - Context length: {config.block_size}")
    print(f"  - Vocabulary size: {config.vocab_size}")
    
    # Load large conversational data
    data_path = os.path.join('datasets', 'large_conversation.txt')
    if not os.path.exists(data_path):
        print(f"Large conversation dataset not found: {data_path}")
        return
    
    print(f"\nLoading large conversation data from: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Data size: {len(text):,} characters")
    
    # Create tokenizer optimized for conversation
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    
    # Encode the data
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Encoded data: {len(data):,} tokens")
    
    # Split into train and validation (larger validation set for conversation quality)
    n = int(0.85 * len(data))  # 85/15 split for better validation
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Training tokens: {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")
    
    # Initialize model
    print(f"\nInitializing conversation model...")
    model = GPT(config)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer optimized for conversation learning
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=0.01,    # Good regularization for conversations
        betas=(0.9, 0.95)     # Good momentum for language modeling
    )
    
    print(f"\nStarting conversation training for {max_iters:,} iterations...")
    print("-" * 45)
    
    # Training variables
    best_val_loss = float('inf')
    start_time = time.time()
    patience = 0
    max_patience = 8   # Reasonable patience for larger dataset
    
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
                patience = 0
                
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'tokenizer': tokenizer,
                    'iter': iter_num,
                    'best_val_loss': best_val_loss,
                    'train_loss': losses['train']
                }
                
                # Save to models directory
                os.makedirs('models', exist_ok=True)
                torch.save(checkpoint, 'models/pico_gpt_conversation.pt')
                print(f"  -> Saved new best model (val loss: {best_val_loss:.4f})")
            else:
                patience += 1
                if patience >= max_patience:
                    print(f"  -> Early stopping after {patience} evaluations without improvement")
                    break
        
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
    
    total_time = time.time() - start_time
    print(f"\n*** Conversation training completed! ***")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved as: models/pico_gpt_conversation.pt")
    
    # Test conversation generation
    print(f"\n*** Testing conversation generation... ***")
    model.eval()
    
    conversation_prompts = [
        "Human: Hello, how are you?",
        "Human: Can you help me with something?",
        "Human: What's your favorite color?",
        "Human: I'm feeling a bit stressed today",
        "Human: Tell me about yourself",
        "Human: Good morning!",
        "Human: Thanks for your help",
        "Human: What can you do?",
    ]
    
    for prompt in conversation_prompts:
        print(f"\n{prompt}")
        context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            generated = model.generate(
                context, 
                max_new_tokens=30,   # Even shorter for testing
                temperature=0.7,     # Lower temperature for coherence
                top_k=20            # More focused responses
            )
        
        generated_text = tokenizer.decode(generated[0].tolist())
        response = generated_text[len(prompt):].strip()
        if response:
            print(f"Assistant: {response}")
        else:
            print(f"Assistant: [No response generated]")
    
    print(f"\n*** Conversation model ready! ***")
    print(f"Use 'python cli/cli_fast.py --model models/pico_gpt_conversation.pt' to chat!")


if __name__ == "__main__":
    train_conversation_model()
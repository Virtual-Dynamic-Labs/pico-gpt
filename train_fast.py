import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import time
import pickle
import os
import math
from pico_gpt import GPT, GPTConfig
from fast_tokenizer import GPT2LikeTokenizer


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


def train_fast_model():
    print("FAST GPT Training - Optimized for Speed & Quality")
    print("=" * 60)
    
    # OPTIMIZED Configuration for FAST training
    config = GPTConfig()
    config.block_size = 128          # Reduced for speed
    config.vocab_size = 8000         # Match new tokenizer
    config.n_layer = 6               # Good balance of speed/quality
    config.n_head = 8                # Efficient attention
    config.n_embd = 384              # Reasonable size
    config.dropout = 0.0             # Disable dropout for speed
    config.bias = False              # Disable bias for speed
    
    # AGGRESSIVE training hyperparameters for speed
    batch_size = 32                  # Larger batches for efficiency
    learning_rate = 6e-4             # Higher LR for faster convergence
    max_iters = 2000                 # Much fewer iterations
    eval_interval = 100              # Frequent evaluation
    eval_iters = 20                  # Quick evaluation
    log_interval = 25                # Frequent logging
    
    # Simple learning rate (no complex schedule)
    weight_decay = 0.01              # Light regularization
    grad_clip = 1.0                  # Gradient clipping
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load FAST tokenizer
    tokenizer_path = "fast_tokenizer_gpt2_8000.pkl"
    if not os.path.exists(tokenizer_path):
        print("[ERROR] Fast tokenizer not found! Run: python fast_tokenizer.py")
        return
    
    print(f"Loading optimized tokenizer: {tokenizer_path}")
    tokenizer = GPT2LikeTokenizer()
    tokenizer.load(tokenizer_path)
    
    # Load and tokenize data EFFICIENTLY
    data_file = "data/combined_literature.txt"
    if not os.path.exists(data_file):
        print(f"[ERROR] Training data not found: {data_file}")
        return
    
    print(f"Loading training data: {data_file}")
    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Use only a portion for FAST training
    text = text[:2_000_000]  # Use first 2M characters for speed
    print(f"Using {len(text):,} characters for fast training")
    
    # Tokenize efficiently
    print("Tokenizing data...")
    start_time = time.time()
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    tokenize_time = time.time() - start_time
    
    print(f"Tokenized in {tokenize_time:.2f}s: {len(data):,} tokens")
    print(f"Compression ratio: {len(text)/len(data):.2f}x")
    
    # Split data
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")
    
    # Initialize model
    print(f"\nInitializing {config.n_layer}-layer model...")
    model = GPT(config)
    model.to(device)
    
    # Enable optimizations (disabled torch.compile due to triton issues)
    # if device == 'cuda':
    #     model = torch.compile(model)  # PyTorch 2.0 optimization
    #     print("[SUCCESS] Model compiled with torch.compile")
    
    total_params = model.get_num_params()
    print(f"Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Optimizer - AdamW with optimized settings
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        betas=(0.9, 0.95), 
        weight_decay=weight_decay,
        fused=True if device == 'cuda' else False  # Fused optimizer for speed
    )
    
    # Training loop
    print(f"\n[TRAINING] Starting FAST training: {max_iters} iterations")
    print(f"Batch size: {batch_size}, Block size: {config.block_size}")
    print("-" * 60)
    
    start_time = time.time()
    best_val_loss = float('inf')
    tokens_per_sec = 0
    
    for iter in range(max_iters):
        # Evaluate periodically
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, eval_iters, batch_size, config.block_size, device)
            elapsed = time.time() - start_time
            
            print(f"iter {iter:4d}: train {losses['train']:.4f}, val {losses['val']:.4f}, "
                  f"time {elapsed:.1f}s, {tokens_per_sec:.0f} tok/s")
            
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'tokenizer': tokenizer,
                    'iter': iter,
                    'best_val_loss': best_val_loss,
                    'train_loss': losses['train']
                }
                torch.save(checkpoint, 'pico_gpt_fast.pt')
        
        # Get batch
        xb, yb = get_batch(train_data, batch_size, config.block_size, device)
        
        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Calculate speed
        if iter > 0 and iter % log_interval == 0:
            elapsed = time.time() - start_time
            tokens_processed = iter * batch_size * config.block_size
            tokens_per_sec = tokens_processed / elapsed
            
            print(f"iter {iter:4d}: loss {loss.item():.4f}, {tokens_per_sec:.0f} tok/s")
    
    # Final results
    total_time = time.time() - start_time
    final_tokens_per_sec = (max_iters * batch_size * config.block_size) / total_time
    
    print(f"\n[SUCCESS] FAST Training Complete!")
    print(f"Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Speed: {final_tokens_per_sec:.0f} tokens/sec")
    print(f"Final loss: {loss.item():.4f}")
    print(f"Best val loss: {best_val_loss:.4f}")
    
    # Generate sample text to test quality
    print(f"\n[TESTING] Generation Quality")
    print("-" * 40)
    
    model.eval()
    test_prompts = [
        "Hello, how are you?",
        "Once upon a time",
        "The weather today is",
        "I think that"
    ]
    
    for prompt in test_prompts:
        context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            generated = model.generate(
                context, 
                max_new_tokens=50, 
                temperature=0.8, 
                top_k=20
            )
        
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated_text}'")
        print()
    
    # Memory summary
    if device == 'cuda':
        memory_used = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak GPU memory: {memory_used:.1f} MB")
    
    print(f"[SUCCESS] Model saved as: pico_gpt_fast.pt")
    return tokenizer, model


if __name__ == "__main__":
    train_fast_model()
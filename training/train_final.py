import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import time
import pickle
import os
import math
import sys
sys.path.append('..')
from src.pico_gpt import GPT, GPTConfig
from src.fast_tokenizer import GPT2LikeTokenizer


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


def train_final_conversation_model():
    print("FINAL Conversation GPT Training - Clean & Fast")
    print("=" * 60)
    
    # OPTIMIZED Configuration for conversation
    config = GPTConfig()
    config.block_size = 128          # Shorter for quick conversations
    config.vocab_size = 8000         # Match tokenizer
    config.n_layer = 6               # Good balance of quality/speed
    config.n_head = 6                # Attention heads
    config.n_embd = 384              # Embedding size
    config.dropout = 0.0             # No dropout for speed
    config.bias = False              # No bias for efficiency
    
    # ULTRA-FAST training parameters
    batch_size = 8                   # Small batch for small dataset
    learning_rate = 1e-3             # Higher learning rate
    max_iters = 1000                 # Quick training
    eval_interval = 100              # Regular evaluation
    eval_iters = 20                  # Quick evaluation
    log_interval = 50                # Frequent logging
    
    weight_decay = 0.0               # No regularization
    grad_clip = 1.0                  # Gradient clipping
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load tokenizer
    tokenizer_path = "../datasets/fast_tokenizer_gpt2_8000.pkl"
    if not os.path.exists(tokenizer_path):
        print("[ERROR] Fast tokenizer not found!")
        return
    
    print(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = GPT2LikeTokenizer()
    tokenizer.load(tokenizer_path)
    
    # Load CLEAN conversation data
    data_file = "../datasets/clean_conversation_data.txt"
    if not os.path.exists(data_file):
        print(f"[ERROR] Clean conversation data not found: {data_file}")
        print("Run: python create_clean_conversation_data.py")
        return
    
    print(f"Loading clean conversation data: {data_file}")
    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Repeat the data to have enough for training
    text = text * 100  # Repeat 100 times for more training data
    print(f"Using {len(text):,} characters of clean conversation data")
    
    # Tokenize
    print("Tokenizing clean conversation data...")
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
    print(f"\nInitializing model ({config.n_layer} layers, {config.n_embd} dim)...")
    model = GPT(config)
    model.to(device)
    
    total_params = model.get_num_params()
    print(f"Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        betas=(0.9, 0.95), 
        weight_decay=weight_decay
    )
    
    # Training loop
    print(f"\n[TRAINING] Final conversation model: {max_iters} iterations")
    print(f"Batch size: {batch_size}, Block size: {config.block_size}")
    print("-" * 60)
    
    start_time = time.time()
    best_val_loss = float('inf')
    
    for iter in range(max_iters):
        # Evaluate periodically
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, eval_iters, batch_size, config.block_size, device)
            elapsed = time.time() - start_time
            
            print(f"iter {iter:4d}: train {losses['train']:.4f}, val {losses['val']:.4f}, time {elapsed:.1f}s")
            
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
                torch.save(checkpoint, '../models/pico_gpt_final.pt')
                print(f"    [SAVED] New best model (val loss: {best_val_loss:.4f})")
        
        # Training step
        xb, yb = get_batch(train_data, batch_size, config.block_size, device)
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Log progress
        if iter % log_interval == 0:
            print(f"iter {iter:4d}: loss {loss.item():.4f}")
    
    # Final results
    total_time = time.time() - start_time
    print(f"\n[SUCCESS] Final Training Complete!")
    print(f"Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Final loss: {loss.item():.4f}")
    print(f"Best val loss: {best_val_loss:.4f}")
    
    # Test the final model
    print(f"\n[TESTING] Final Conversation Quality")
    print("-" * 40)
    
    model.eval()
    test_prompts = [
        "Human: Hello",
        "Human: Hi, my name is Alice", 
        "Human: How are you?",
        "Human: What can you do?",
        "Human: Thank you",
        "Human: I'm feeling sad",
        "Human: Goodbye"
    ]
    
    for prompt in test_prompts:
        context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            generated = model.generate(
                context, 
                max_new_tokens=20, 
                temperature=0.8, 
                top_k=10
            )
        
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"Input: '{prompt}'")
        print(f"Output: '{generated_text}'")
        print()
    
    # Memory usage
    if device == 'cuda':
        memory_used = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak GPU memory: {memory_used:.1f} MB")
    
    print(f"[SUCCESS] Final model saved as: ../models/pico_gpt_final.pt")
    return model, tokenizer


if __name__ == "__main__":
    train_final_conversation_model()
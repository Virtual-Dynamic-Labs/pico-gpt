import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import time
import pickle
import os
import math
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
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def train_large_model():
    print("Training Large Pico GPT Model")
    print("=" * 50)
    
    # Enhanced Configuration for RTX 3080 Ti (12GB VRAM)
    # These settings are optimized for 12GB VRAM while maximizing model capability
    config = GPTConfig()
    config.block_size = 256          # Reduced sequence length for memory efficiency
    config.vocab_size = 1000         # Match tokenizer vocab size
    config.n_layer = 8               # Reduced layers for stable training
    config.n_head = 8                # Reduced heads
    config.n_embd = 512              # Reduced embedding dimension
    config.dropout = 0.1             # Dropout rate
    config.bias = True               # Use bias in linear layers
    
    # Training hyperparameters
    batch_size = 16                  # Increased for better training
    learning_rate = 1e-3             # Higher learning rate for faster convergence
    max_iters = 5000                 # Reasonable training iterations
    eval_interval = 200              # Evaluate more frequently
    eval_iters = 50                  # Fewer evaluation iterations
    log_interval = 50                # Log more frequently
    save_interval = 500              # Save more frequently
    
    # Learning rate schedule
    warmup_iters = 200               # Reduced warmup iterations
    lr_decay_iters = 4000            # Decay learning rate over these iterations
    min_lr = 1e-4                    # Minimum learning rate
    
    # Gradient clipping
    grad_clip = 1.0
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load training data
    data_file = None
    for filename in ['training_data.txt', 'data/combined_literature.txt', 'shakespeare.txt']:
        if os.path.exists(filename):
            data_file = filename
            break
    
    if not data_file:
        print("No training data found! Please run: python download_dataset.py")
        return
    
    print(f"Loading data from: {data_file}")
    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Dataset size: {len(text):,} characters ({len(text)/1e6:.1f}M)")
    
    # Create tokenizer matching config vocab size
    print("Creating tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    
    # Tokenize the data
    print("Tokenizing data...")
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Tokenized data: {len(data):,} tokens")
    
    # Split into train and validation
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")
    
    # Initialize model
    print("\nInitializing model...")
    model = GPT(config)
    model.to(device)
    
    # Count parameters
    total_params = model.get_num_params()
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Estimate memory usage
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Test memory usage with a forward pass
        test_x, test_y = get_batch(train_data, batch_size, config.block_size, device)
        logits, loss = model(test_x, test_y)
        
        memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"Estimated memory usage: {memory_used:.1f} MB")
        
        torch.cuda.empty_cache()
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    
    # Verify we have enough data
    min_required_tokens = batch_size * config.block_size * 10
    if len(train_data) < min_required_tokens:
        print(f"WARNING: Training data may be too small ({len(train_data)} tokens)")
        print(f"Recommended minimum: {min_required_tokens} tokens")
    
    # Training loop
    print(f"\nStarting training for {max_iters} iterations...")
    print(f"Batch size: {batch_size}, Block size: {config.block_size}")
    print(f"Vocab size: {config.vocab_size}, Model size: {config.n_layer} layers")
    print("-" * 70)
    
    start_time = time.time()
    best_val_loss = float('inf')
    
    # Test initial loss
    initial_loss = estimate_loss(model, train_data, val_data, 10, batch_size, config.block_size, device)
    print(f"Initial train loss: {initial_loss['train']:.4f}, val loss: {initial_loss['val']:.4f}")
    print("-" * 70)
    
    for iter in range(max_iters):
        # Determine learning rate
        lr = get_lr(iter, warmup_iters, learning_rate, lr_decay_iters, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate the loss on train/val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, eval_iters, batch_size, config.block_size, device)
            
            elapsed = time.time() - start_time
            print(f"step {iter:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
                  f"lr {lr:.2e}, time {elapsed:.1f}s")
            
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'tokenizer': tokenizer,
                    'iter': iter + 1,  # Save the next iteration number
                    'best_val_loss': best_val_loss,
                    'train_loss': losses['train']
                }
                torch.save(checkpoint, 'pico_gpt_large_best.pt')
                print(f"    Saved new best model (val loss: {best_val_loss:.4f})")
        
        # Sample a batch of data
        xb, yb = get_batch(train_data, batch_size, config.block_size, device)
        
        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Log progress
        if iter % log_interval == 0:
            print(f"iter {iter:5d}: loss {loss.item():.4f}, lr {lr:.2e}")
        
        # Save checkpoint
        if iter % save_interval == 0 and iter > 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'tokenizer': tokenizer,
                'iter': iter,
                'train_loss': loss.item()
            }
            torch.save(checkpoint, f'pico_gpt_large_checkpoint_{iter}.pt')
            print(f"Saved checkpoint at iteration {iter}")
    
    # Final save
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'tokenizer': tokenizer,
        'iter': max_iters,
        'final_loss': loss.item()
    }
    torch.save(final_checkpoint, 'pico_gpt_large_final.pt')
    
    total_time = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Final loss: {loss.item():.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Generate sample text
    print("\n" + "=" * 50)
    print("SAMPLE GENERATION")
    print("=" * 50)
    
    model.eval()
    prompts = ["The", "Once upon a time", "In a", "The quick brown fox"]
    
    for prompt in prompts:
        context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            generated = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=20)
        
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{generated_text}'")
        print("-" * 50)
    
    # Memory summary
    if device == 'cuda':
        print(f"\nGPU Memory Summary:")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
        print(f"Max memory reserved: {torch.cuda.max_memory_reserved() / 1024**2:.1f} MB")


if __name__ == "__main__":
    train_large_model()
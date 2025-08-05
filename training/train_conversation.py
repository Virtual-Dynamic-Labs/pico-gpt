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


def train_conversation_model():
    print("CONVERSATION GPT Training - Optimized for Chat")
    print("=" * 60)
    
    # CONVERSATION-OPTIMIZED Configuration
    config = GPTConfig()
    config.block_size = 256          # Longer context for conversations
    config.vocab_size = 8000         # Match tokenizer
    config.n_layer = 8               # Deeper for better understanding
    config.n_head = 8                # Multi-head attention
    config.n_embd = 512              # Good embedding size
    config.dropout = 0.1             # Light dropout for regularization
    config.bias = False              # No bias for efficiency
    
    # FAST training hyperparameters
    batch_size = 16                  # Balanced batch size
    learning_rate = 3e-4             # Conservative learning rate
    max_iters = 3000                 # More iterations for conversation quality
    eval_interval = 200              # Regular evaluation
    eval_iters = 50                  # Quick evaluation
    log_interval = 50                # Frequent logging
    
    # Learning rate schedule
    warmup_iters = 300               # Warmup period
    lr_decay_iters = 2500            # Decay over most of training
    min_lr = 3e-5                    # Minimum learning rate
    
    weight_decay = 0.01              # Light regularization
    grad_clip = 1.0                  # Gradient clipping
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load conversation-optimized tokenizer
    tokenizer_path = "fast_tokenizer_gpt2_8000.pkl"
    if not os.path.exists(tokenizer_path):
        print("[ERROR] Fast tokenizer not found! Run: python fast_tokenizer.py")
        return
    
    print(f"Loading optimized tokenizer: {tokenizer_path}")
    tokenizer = GPT2LikeTokenizer()
    tokenizer.load(tokenizer_path)
    
    # Load CONVERSATION data
    conv_data_files = ["simple_conversation_data.txt", "conversation_data.txt"]
    conversation_text = ""
    
    for data_file in conv_data_files:
        if os.path.exists(data_file):
            print(f"Loading conversation data: {data_file}")
            with open(data_file, 'r', encoding='utf-8') as f:
                conversation_text += f.read() + "\n\n"
    
    if not conversation_text:
        print("[ERROR] No conversation data found!")
        print("Run: python create_conversation_data.py")
        return
    
    print(f"Using {len(conversation_text):,} characters of conversation data")
    
    # Tokenize conversation data
    print("Tokenizing conversation data...")
    start_time = time.time()
    data = torch.tensor(tokenizer.encode(conversation_text), dtype=torch.long)
    tokenize_time = time.time() - start_time
    
    print(f"Tokenized in {tokenize_time:.2f}s: {len(data):,} tokens")
    print(f"Compression ratio: {len(conversation_text)/len(data):.2f}x")
    
    # Split data
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")
    
    # Initialize model
    print(f"\nInitializing conversation model ({config.n_layer} layers)...")
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
    
    # Learning rate scheduler function
    def get_lr(it):
        # Linear warmup
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # Cosine decay
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)
    
    # Training loop
    print(f"\n[TRAINING] Conversation model: {max_iters} iterations")
    print(f"Batch size: {batch_size}, Block size: {config.block_size}")
    print("-" * 60)
    
    start_time = time.time()
    best_val_loss = float('inf')
    tokens_per_sec = 0
    
    for iter in range(max_iters):
        # Set learning rate
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate periodically
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, eval_iters, batch_size, config.block_size, device)
            elapsed = time.time() - start_time
            
            print(f"iter {iter:4d}: train {losses['train']:.4f}, val {losses['val']:.4f}, "
                  f"lr {lr:.2e}, time {elapsed:.1f}s, {tokens_per_sec:.0f} tok/s")
            
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
                torch.save(checkpoint, 'pico_gpt_conversation.pt')
                print(f"    [SAVED] New best model (val loss: {best_val_loss:.4f})")
        
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
            
            print(f"iter {iter:4d}: loss {loss.item():.4f}, lr {lr:.2e}, {tokens_per_sec:.0f} tok/s")
    
    # Final results
    total_time = time.time() - start_time
    final_tokens_per_sec = (max_iters * batch_size * config.block_size) / total_time
    
    print(f"\n[SUCCESS] Conversation Training Complete!")
    print(f"Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Speed: {final_tokens_per_sec:.0f} tokens/sec")
    print(f"Final loss: {loss.item():.4f}")
    print(f"Best val loss: {best_val_loss:.4f}")
    
    # Test conversation quality
    print(f"\n[TESTING] Conversation Quality")
    print("-" * 40)
    
    model.eval()
    conversation_tests = [
        "Human: Hello",
        "Human: Hi, my name is Alice",
        "Human: How are you?",
        "Human: What can you do?",
        "Human: Tell me about yourself",
        "Human: I'm feeling sad",
        "Human: Thank you"
    ]
    
    for prompt in conversation_tests:
        context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            generated = model.generate(
                context, 
                max_new_tokens=30, 
                temperature=0.7, 
                top_k=20
            )
        
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"Test: '{prompt}'")
        print(f"Response: '{generated_text}'")
        print()
    
    # Memory summary
    if device == 'cuda':
        memory_used = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak GPU memory: {memory_used:.1f} MB")
    
    print(f"[SUCCESS] Conversation model saved as: pico_gpt_conversation.pt")
    return tokenizer, model


if __name__ == "__main__":
    train_conversation_model()
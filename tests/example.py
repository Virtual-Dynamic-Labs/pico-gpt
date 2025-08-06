#!/usr/bin/env python3
"""
Example usage of pico-GPT
"""

import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.pico_gpt import GPT, GPTConfig
from src.tokenizer import SimpleTokenizer


def create_tiny_gpt():
    """Create a very small GPT model for quick testing"""
    config = GPTConfig()
    config.block_size = 64      # Small context window
    config.vocab_size = 100     # Small vocabulary
    config.n_layer = 2         # Just 2 layers
    config.n_head = 2          # 2 attention heads
    config.n_embd = 64         # Small embedding dimension
    config.dropout = 0.0
    
    model = GPT(config)
    print(f"Created tiny GPT with {model.get_num_params():,} parameters")
    return model, config


def test_forward_pass():
    """Test a forward pass through the model"""
    print("\n=== Testing Forward Pass ===")
    
    model, config = create_tiny_gpt()
    
    # Create some dummy input
    batch_size = 2
    seq_len = 10
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    dummy_targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    logits, loss = model(dummy_input, dummy_targets)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")


def test_generation():
    """Test text generation"""
    print("\n=== Testing Generation ===")
    
    model, config = create_tiny_gpt()
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    
    # Generate from a simple prompt
    prompt = "Hello"
    context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)
    
    print(f"Prompt: '{prompt}'")
    print(f"Encoded prompt: {context.tolist()}")
    
    # Generate (will be random gibberish since model is untrained)
    model.eval()
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=20, temperature=1.0, top_k=10)
    
    generated_text = tokenizer.decode(generated[0].tolist())
    print(f"Generated text: '{generated_text}'")


def test_training_step():
    """Test a single training step"""
    print("\n=== Testing Training Step ===")
    
    model, config = create_tiny_gpt()
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    
    # Create some training data
    text = "Hello world! This is a test. Hello world again!"
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    # Create a batch
    batch_size = 2
    block_size = 16
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    print(f"Training batch shapes - X: {x.shape}, Y: {y.shape}")
    
    # Forward pass
    logits, loss = model(x, y)
    print(f"Initial loss: {loss.item():.4f}")
    
    # Backward pass
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Check loss after one step
    logits, loss = model(x, y)
    print(f"Loss after 1 step: {loss.item():.4f}")


def main():
    print("Pico-GPT Example Usage")
    print("=" * 30)
    
    test_forward_pass()
    test_generation()
    test_training_step()
    
    print("\n=== Model Architecture ===")
    model, config = create_tiny_gpt()
    print(f"Block size: {config.block_size}")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Number of layers: {config.n_layer}")
    print(f"Number of attention heads: {config.n_head}")
    print(f"Embedding dimension: {config.n_embd}")
    print(f"Total parameters: {model.get_num_params():,}")
    
    print("\nTo train a full model, run: python train.py")
    print("To generate text, run: python generate.py --prompt 'Your prompt here'")


if __name__ == "__main__":
    main()
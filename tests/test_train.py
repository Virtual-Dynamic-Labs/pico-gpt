import torch
from pico_gpt import GPT, GPTConfig
from tokenizer import SimpleTokenizer
import time


def quick_training_test():
    """Quick training test with minimal configuration"""
    print("=== Quick Training Test ===")
    
    # Very small configuration for fast testing
    config = GPTConfig()
    config.block_size = 32      # Small context
    config.vocab_size = 100     # Small vocab
    config.n_layer = 2         # Just 2 layers
    config.n_head = 2          # 2 heads
    config.n_embd = 64         # Small embedding
    config.dropout = 0.1
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create sample training text
    sample_text = "Hello world! This is a test. The quick brown fox jumps over the lazy dog. " * 50
    
    # Tokenize
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    data = torch.tensor(tokenizer.encode(sample_text), dtype=torch.long)
    print(f"Training data size: {len(data)} tokens")
    
    # Create model
    model = GPT(config)
    model.to(device)
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Training parameters
    batch_size = 4
    learning_rate = 1e-3
    max_iters = 100
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    start_time = time.time()
    
    for iter in range(max_iters):
        # Sample batch
        ix = torch.randint(len(data) - config.block_size, (batch_size,))
        x = torch.stack([data[i:i+config.block_size] for i in ix]).to(device)
        y = torch.stack([data[i+1:i+config.block_size+1] for i in ix]).to(device)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if iter % 20 == 0:
            print(f"iter {iter:3d}: loss {loss.item():.4f}")
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    # Test generation
    print("\n=== Testing Generation After Training ===")
    model.eval()
    
    prompt = "Hello"
    context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=30, temperature=0.8, top_k=10)
    
    generated_text = tokenizer.decode(generated[0].tolist())
    print(f"Prompt: '{prompt}'")
    print(f"Generated: '{generated_text}'")
    
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = quick_training_test()
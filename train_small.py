import torch
from pico_gpt import GPT, GPTConfig
from tokenizer import SimpleTokenizer


def train_small_model():
    """Train a very small model for testing generation"""
    print("Training small model for testing...")
    
    # Small configuration
    config = GPTConfig()
    config.block_size = 64
    config.vocab_size = 200
    config.n_layer = 3
    config.n_head = 4
    config.n_embd = 128
    config.dropout = 0.1
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create training text with more structure
    sample_text = """Hello world! This is a simple test.
The quick brown fox jumps over the lazy dog.
Python is a great programming language.
Machine learning is fascinating.
Hello there, how are you today?
The weather is nice and sunny.
I love learning new things every day.
Programming is fun and rewarding.
""" * 20  # Repeat for more data
    
    # Tokenize
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    data = torch.tensor(tokenizer.encode(sample_text), dtype=torch.long)
    
    # Split data
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Training data: {len(train_data)} tokens")
    print(f"Validation data: {len(val_data)} tokens")
    
    # Create model
    model = GPT(config)

    if torch.cuda.is_available():
        print("Using CUDA for training")
    else:
        print("Using CPU for training")
    
    model.to(device)
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Training settings
    batch_size = 8
    learning_rate = 3e-4
    max_iters = 500
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    
    for iter in range(max_iters):
        # Sample batch
        ix = torch.randint(len(train_data) - config.block_size, (batch_size,))
        x = torch.stack([train_data[i:i+config.block_size] for i in ix]).to(device)
        y = torch.stack([train_data[i+1:i+config.block_size+1] for i in ix]).to(device)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if iter % 100 == 0:
            print(f"iter {iter:3d}: loss {loss.item():.4f}")
    
    # Final validation
    model.eval()
    with torch.no_grad():
        ix = torch.randint(len(val_data) - config.block_size, (4,))
        x = torch.stack([val_data[i:i+config.block_size] for i in ix]).to(device)
        y = torch.stack([val_data[i+1:i+config.block_size+1] for i in ix]).to(device)
        logits, val_loss = model(x, y)
        print(f"Final validation loss: {val_loss.item():.4f}")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer': tokenizer
    }, 'pico_gpt_model.pt')
    
    print("Model saved as 'pico_gpt_model.pt'")
    
    # Test generation
    print("\n=== Sample Generation ===")
    model.eval()
    
    test_prompts = ["Hello", "The", "Python"]
    
    for prompt in test_prompts:
        context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            generated = model.generate(context, max_new_tokens=20, temperature=0.8, top_k=10)
        
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"'{prompt}' -> '{generated_text}'")


if __name__ == "__main__":
    train_small_model()
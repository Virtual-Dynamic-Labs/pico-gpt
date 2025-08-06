import torch
import torch.nn as nn
import time
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pico_gpt import GPT, GPTConfig
from src.tokenizer import SimpleTokenizer


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def benchmark_training(device_name, num_iters=50):
    print(f"\n=== Benchmarking on {device_name.upper()} ===")
    
    # Force device
    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load the large trained model
    try:
        checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pico_gpt_large_best.pt')
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        config = checkpoint['config']
        model = GPT(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        tokenizer = checkpoint['tokenizer']
        
        print(f"Loaded large model with {model.get_num_params():,} parameters")
        
    except FileNotFoundError:
        print("Large model not found, falling back to small model")
        config = GPTConfig()
        config.block_size = 128
        config.vocab_size = 1000
        config.n_layer = 4
        config.n_head = 4
        config.n_embd = 256
        config.dropout = 0.1
        
        model = GPT(config)
        model.to(device)
        tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    
    # Create sample data
    sample_text = """Hello world! This is a benchmark test for comparing CUDA vs CPU performance.
    The quick brown fox jumps over the lazy dog. Python is great for machine learning.
    Neural networks are fascinating and powerful tools for artificial intelligence.
    Training on GPU should be significantly faster than CPU for deep learning models.
    Once upon a time in a land far away, there lived a wise old wizard who knew the secrets of deep learning.
    """ * 20  # Repeat for more data
    
    # Tokenize with appropriate vocab size
    data = torch.tensor(tokenizer.encode(sample_text), dtype=torch.long)
    
    # Training settings
    batch_size = 16
    learning_rate = 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Training for {num_iters} iterations with batch size {batch_size}")
    
    # Warm up (first few iterations can be slower)
    model.train()
    for _ in range(5):
        xb, yb = get_batch(data, batch_size, config.block_size, device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Clear cache for accurate timing
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Benchmark training
    start_time = time.time()
    losses = []
    
    for iter in range(num_iters):
        # Sample batch
        xb, yb = get_batch(data, batch_size, config.block_size, device)
        
        # Forward pass
        logits, loss = model(xb, yb)
        losses.append(loss.item())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress every 10 iterations
        if (iter + 1) % 10 == 0:
            print(f"  Iteration {iter + 1}/{num_iters}, Loss: {loss.item():.4f}")
    
    # Ensure all operations complete
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate metrics
    avg_loss = np.mean(losses)
    time_per_iter = total_time / num_iters
    samples_per_second = (num_iters * batch_size) / total_time
    
    print(f"\n{device_name.upper()} Results:")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Time per iteration: {time_per_iter*1000:.2f} ms")
    print(f"  Samples per second: {samples_per_second:.1f}")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    
    return {
        'device': device_name,
        'total_time': total_time,
        'time_per_iter': time_per_iter,
        'samples_per_second': samples_per_second,
        'avg_loss': avg_loss,
        'final_loss': losses[-1]
    }


def main():
    print("CUDA vs CPU Training Benchmark")
    print("=" * 50)
    
    # Check available devices
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Run benchmarks (reduced iterations for large model)
    num_iterations = 50
    
    # Benchmark CPU
    cpu_results = benchmark_training('cpu', num_iterations)
    
    # Benchmark CUDA (if available)
    if torch.cuda.is_available():
        cuda_results = benchmark_training('cuda', num_iterations)
        
        # Compare results
        print(f"\n{'='*50}")
        print("PERFORMANCE COMPARISON")
        print(f"{'='*50}")
        
        speedup = cpu_results['total_time'] / cuda_results['total_time']
        throughput_improvement = cuda_results['samples_per_second'] / cpu_results['samples_per_second']
        
        print(f"CPU Total Time:    {cpu_results['total_time']:.2f}s")
        print(f"CUDA Total Time:   {cuda_results['total_time']:.2f}s")
        print(f"Speedup:           {speedup:.2f}x faster")
        print(f"")
        print(f"CPU Throughput:    {cpu_results['samples_per_second']:.1f} samples/sec")
        print(f"CUDA Throughput:   {cuda_results['samples_per_second']:.1f} samples/sec")
        print(f"Improvement:       {throughput_improvement:.2f}x more throughput")
        print(f"")
        print(f"Time per iteration:")
        print(f"  CPU:  {cpu_results['time_per_iter']*1000:.2f} ms")
        print(f"  CUDA: {cuda_results['time_per_iter']*1000:.2f} ms")
        
        # Memory usage (if CUDA)
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1024**2   # MB
            print(f"")
            print(f"GPU Memory Usage:")
            print(f"  Allocated: {memory_allocated:.1f} MB")
            print(f"  Reserved:  {memory_reserved:.1f} MB")
        
    else:
        print("\nCUDA not available - only CPU benchmark completed")


if __name__ == "__main__":
    main()
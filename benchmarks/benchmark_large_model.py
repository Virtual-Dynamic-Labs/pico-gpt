import torch
import time
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append('..')
from src.pico_gpt import GPT


def benchmark_inference(device_name, num_runs=50):
    print(f"\n=== Benchmarking Large Model Inference on {device_name.upper()} ===")
    
    # Force device
    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load the large trained model
    try:
        checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'pico_gpt_final.pt')
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        config = checkpoint['config']
        model = GPT(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        tokenizer = checkpoint['tokenizer']
        
        print(f"Loaded large model with {model.get_num_params():,} parameters")
        print(f"Config: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} embedding dim")
        
    except FileNotFoundError:
        print("ERROR: Large model 'pico_gpt_large_best.pt' not found!")
        return None
    
    # Test prompts for inference
    test_prompts = [
        "Once upon a time",
        "The quick brown fox",
        "In a land far away",
        "She walked through the forest"
    ]
    
    # Warmup runs
    print("Warming up...")
    for prompt in test_prompts[:2]:
        context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            model.generate(context, max_new_tokens=20, temperature=0.7, top_k=20)
    
    # Clear cache for accurate timing
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Benchmark inference
    print(f"Running {num_runs} inference tests...")
    times = []
    tokens_generated = 50  # tokens per generation
    
    start_time = time.time()
    
    for run in range(num_runs):
        prompt = test_prompts[run % len(test_prompts)]
        context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        run_start = time.time()
        with torch.no_grad():
            generated = model.generate(
                context, 
                max_new_tokens=tokens_generated, 
                temperature=0.7, 
                top_k=20
            )
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        run_time = time.time() - run_start
        times.append(run_time)
        
        if (run + 1) % 10 == 0:
            print(f"  Completed {run + 1}/{num_runs} runs")
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)
    tokens_per_second = tokens_generated / avg_time
    total_tokens = num_runs * tokens_generated
    
    print(f"\n{device_name.upper()} Inference Results:")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Average time per generation: {avg_time*1000:.2f} ms")
    print(f"  Min time: {min_time*1000:.2f} ms")
    print(f"  Max time: {max_time*1000:.2f} ms")
    print(f"  Std deviation: {std_time*1000:.2f} ms")
    print(f"  Tokens per second: {tokens_per_second:.1f}")
    print(f"  Total tokens generated: {total_tokens:,}")
    
    # Memory usage
    if device.type == 'cuda':
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2   # MB
        print(f"  GPU Memory - Allocated: {memory_allocated:.1f} MB, Reserved: {memory_reserved:.1f} MB")
    
    return {
        'device': device_name,
        'total_time': total_time,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'tokens_per_second': tokens_per_second,
        'memory_allocated': memory_allocated if device.type == 'cuda' else 0
    }


def main():
    print("Large Model CUDA vs CPU Inference Benchmark")
    print("=" * 60)
    
    # Check available devices
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    num_runs = 50
    
    # Benchmark CPU
    cpu_results = benchmark_inference('cpu', num_runs)
    
    # Benchmark CUDA (if available)
    if torch.cuda.is_available() and cpu_results:
        cuda_results = benchmark_inference('cuda', num_runs)
        
        if cuda_results:
            # Compare results
            print(f"\n{'='*60}")
            print("PERFORMANCE COMPARISON")
            print(f"{'='*60}")
            
            speedup = cpu_results['avg_time'] / cuda_results['avg_time']
            throughput_improvement = cuda_results['tokens_per_second'] / cpu_results['tokens_per_second']
            
            print(f"Average Generation Time:")
            print(f"  CPU:  {cpu_results['avg_time']*1000:.2f} ms")
            print(f"  CUDA: {cuda_results['avg_time']*1000:.2f} ms")
            print(f"  Speedup: {speedup:.2f}x faster")
            print(f"")
            print(f"Throughput (tokens/sec):")
            print(f"  CPU:  {cpu_results['tokens_per_second']:.1f}")
            print(f"  CUDA: {cuda_results['tokens_per_second']:.1f}")
            print(f"  Improvement: {throughput_improvement:.2f}x")
            print(f"")
            print(f"Memory Usage:")
            print(f"  CUDA Allocated: {cuda_results['memory_allocated']:.1f} MB")
    else:
        print("\nCUDA not available or model loading failed - only CPU benchmark completed")


if __name__ == "__main__":
    main()
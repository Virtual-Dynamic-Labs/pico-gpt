import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pico_gpt import GPT
import time


def test_large_model():
    print("Testing Large Pico GPT Model")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load the best model
    try:
        checkpoint = torch.load('pico_gpt_large_best.pt', map_location=device, weights_only=False)
        
        config = checkpoint['config']
        model = GPT(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        tokenizer = checkpoint['tokenizer']
        
        print(f"Model loaded successfully!")
        print(f"Model parameters: {model.get_num_params():,} ({model.get_num_params()/1e6:.2f}M)")
        print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
        print(f"Training iteration: {checkpoint.get('iter', 'N/A')}")
        
    except FileNotFoundError:
        print("No trained model found. Please run: python train_large.py")
        return
    
    # Test generation with various prompts
    print(f"\n" + "=" * 50)
    print("SAMPLE GENERATIONS")
    print("=" * 50)
    
    test_prompts = [
        "Once upon a time",
        "The quick brown fox",
        "In a land far away",
        "Elizabeth",
        "It was the best of times",
        "The man",
        "She walked"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 30)
        
        # Encode prompt
        context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate with different settings
        settings = [
            {"temp": 0.7, "top_k": 20, "tokens": 80, "name": "Balanced"},
            {"temp": 0.9, "top_k": 40, "tokens": 60, "name": "Creative"},
            {"temp": 0.5, "top_k": 10, "tokens": 100, "name": "Conservative"}
        ]
        
        for setting in settings[:1]:  # Just use balanced for speed
            with torch.no_grad():
                start_time = time.time()
                generated = model.generate(
                    context, 
                    max_new_tokens=setting["tokens"], 
                    temperature=setting["temp"], 
                    top_k=setting["top_k"]
                )
                generation_time = time.time() - start_time
            
            generated_text = tokenizer.decode(generated[0].tolist())
            tokens_per_sec = setting["tokens"] / generation_time
            
            print(f"{setting['name']}: {generated_text}")
            print(f"Speed: {tokens_per_sec:.1f} tokens/sec")
    
    # Memory usage
    if device == 'cuda':
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        print(f"\nGPU Memory Used: {memory_allocated:.1f} MB")
    
    print(f"\nModel testing complete!")


if __name__ == "__main__":
    test_large_model()
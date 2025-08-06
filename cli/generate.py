import torch
import sys
import os
# Get the absolute path to the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Create compatibility layer for old module references
import src.pico_gpt as pico_gpt
import src.tokenizer as tokenizer
import src.fast_tokenizer as fast_tokenizer
sys.modules['pico_gpt'] = pico_gpt
sys.modules['tokenizer'] = tokenizer
sys.modules['fast_tokenizer'] = fast_tokenizer

from src.pico_gpt import GPT
import argparse


def load_model(model_path):
    # Try to find the model file in common locations
    potential_paths = [
        model_path,  # Original path as specified
        os.path.join('models', model_path),  # Try in models/ directory
        os.path.join('models', os.path.basename(model_path))  # Try basename in models/
    ]
    
    # Find the first path that exists
    actual_model_path = None
    for path in potential_paths:
        if os.path.exists(path):
            actual_model_path = path
            break
    
    if not actual_model_path:
        raise FileNotFoundError(f"Model file not found: {model_path}. Tried locations: {potential_paths}")
    
    # Note: Using weights_only=False because we need to load the full model objects
    # including config and tokenizer. Only use with trusted model files.
    checkpoint = torch.load(actual_model_path, map_location='cpu', weights_only=False)
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    tokenizer = checkpoint['tokenizer']
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=0.8, top_k=10, device='cpu'):
    model.eval()
    model.to(device)
    
    # Encode the prompt
    context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            context, 
            max_new_tokens=max_tokens, 
            temperature=temperature, 
            top_k=top_k
        )
    
    # Decode and return
    generated_text = tokenizer.decode(generated[0].tolist())
    return generated_text


def main():
    parser = argparse.ArgumentParser(description='Generate text with pico-GPT')
    parser.add_argument('--model', default='models/pico_gpt_large_best.pt', help='Path to model checkpoint')
    parser.add_argument('--prompt', default='Hello', help='Text prompt for generation')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=10, help='Top-k sampling parameter')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        model, tokenizer = load_model(args.model)
        print(f"Model loaded from {args.model}")
        
        generated_text = generate_text(
            model, tokenizer, args.prompt, 
            args.max_tokens, args.temperature, args.top_k, device
        )
        
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated: {generated_text}")
        
    except FileNotFoundError:
        print(f"Model file {args.model} not found. Please train a model first using train.py")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Upload Pico GPT Conversational Model to Hugging Face Hub
"""

import torch
import os
from huggingface_hub import HfApi, create_repo, upload_file
from pathlib import Path

def create_model_card():
    """Create a README.md model card for Hugging Face"""
    
    # Load model info
    checkpoint = torch.load('models/pico_gpt_conversation.pt', map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    model_card = f"""---
license: mit
language: en
tags:
- conversational-ai
- gpt
- pytorch
- text-generation
- chatbot
widget:
- text: "Human: Hello, how are you?"
  example_title: "Greeting"
- text: "Human: Can you help me with something?"
  example_title: "Request for help"
- text: "Human: What's your favorite color?"
  example_title: "Question"
---

# Pico GPT Conversational Model

A small but effective conversational AI model trained for natural dialogue interactions.

## Model Details

- **Model Type**: GPT-style autoregressive language model
- **Parameters**: {sum(p.numel() for p in checkpoint['model_state_dict'].values()):,}
- **Vocabulary Size**: {config.vocab_size:,}
- **Context Length**: {config.block_size} tokens
- **Architecture**: {config.n_layer} layers, {config.n_head} attention heads, {config.n_embd} embedding dimension
- **Training Loss**: {checkpoint['best_val_loss']:.4f}
- **Training Iterations**: {checkpoint.get('iter', 'Unknown')}

## Usage

```python
import torch
from src.pico_gpt import GPT, GPTConfig
from src.tokenizer import SimpleTokenizer

# Load the model
checkpoint = torch.load('pico_gpt_conversation.pt', map_location='cpu')
config = checkpoint['config']
tokenizer = checkpoint['tokenizer']

model = GPT(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate response
prompt = "Human: Hello, how are you?\\nAssistant:"
tokens = tokenizer.encode(prompt)
context = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

with torch.no_grad():
    generated = model.generate(context, max_new_tokens=50, temperature=0.7, top_k=15)

response = tokenizer.decode(generated[0].tolist())
print(response)
```

## Training

The model was trained on a curated dataset of conversational exchanges with:
- High-quality human-assistant dialogues
- Natural conversation patterns
- Clear response boundaries using separators
- Repetitive training for pattern learning

## Limitations

- Designed for single-turn conversations (each exchange is independent)
- Limited vocabulary of {config.vocab_size:,} tokens
- Context window of {config.block_size} tokens
- Best suited for short, natural responses

## Example Conversations

**Input**: "Human: Hello, how are you?"
**Output**: "Hello! How are you?"

**Input**: "Human: Can you help me with something?"  
**Output**: "Of course! I'd be happy to help. What do you need?"

**Input**: "Human: What's your favorite color?"
**Output**: "I don't have preferences like humans do, but I find all colors interesting! What's yours?"

## Files Included

- `pico_gpt_conversation.pt` - The trained model checkpoint
- `src/pico_gpt.py` - Model architecture
- `src/tokenizer.py` - Tokenizer implementation  
- `cli/cli_client.py` - Command-line interface
- `run_cli.ps1` - Easy launcher script

## License

MIT License - Feel free to use and modify for your projects!
"""
    
    return model_card

def main():
    """Main upload function"""
    
    print("Preparing Pico GPT Conversational Model for Hugging Face upload...")
    
    # Check if model exists
    model_path = "models/pico_gpt_conversation.pt"
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return
    
    # Create model card
    print("Creating model card...")
    model_card = create_model_card()
    
    # Save model card
    with open("README.md.model", "w", encoding="utf-8") as f:
        f.write(model_card)
    
    print("Model card created: README.md.model")
    print()
    print("Files ready for upload:")
    print(f"   - {model_path} ({os.path.getsize(model_path) / 1024 / 1024:.1f} MB)")
    print("   - README.md.model (model card)")
    print("   - src/pico_gpt.py (model architecture)")
    print("   - src/tokenizer.py (tokenizer)")
    print("   - cli/cli_client.py (interface)")
    print()
    print("Next steps:")
    print("1. Go to https://huggingface.co/new")
    print("2. Create a new model repository")
    print("3. Run the commands below:")
    print()
    print("   # Login to Hugging Face")
    print("   huggingface-cli login")
    print()
    print("   # Upload files (replace YOUR_USERNAME and MODEL_NAME)")
    print("   python -c \"")
    print("from huggingface_hub import upload_file")
    print("repo_id = 'YOUR_USERNAME/pico-gpt-conversational'")
    print("upload_file('models/pico_gpt_conversation.pt', repo_id=repo_id, path_in_repo='pico_gpt_conversation.pt')")
    print("upload_file('README.md.model', repo_id=repo_id, path_in_repo='README.md')")
    print("upload_file('src/pico_gpt.py', repo_id=repo_id, path_in_repo='src/pico_gpt.py')")
    print("upload_file('src/tokenizer.py', repo_id=repo_id, path_in_repo='src/tokenizer.py')")
    print("upload_file('cli/cli_client.py', repo_id=repo_id, path_in_repo='cli/cli_client.py')")
    print("   \"")
    print()
    print("Your conversational AI will then be available at:")
    print("   https://huggingface.co/YOUR_USERNAME/pico-gpt-conversational")

if __name__ == "__main__":
    main()
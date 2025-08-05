# 🤖 Pico-GPT

A minimal, educational implementation of the GPT (Generative Pre-trained Transformer) architecture in PyTorch. This project provides a clean, well-documented codebase for understanding how GPT models work under the hood.

## 🌟 Features

- **Complete GPT Architecture**: Multi-head self-attention, position embeddings, layer normalization
- **Configurable Model Size**: Easily adjust layers, heads, embedding dimensions
- **Training Pipeline**: Full training loop with validation and checkpointing
- **Text Generation**: Autoregressive generation with temperature and top-k sampling
- **Simple Tokenization**: Character-level and basic BPE tokenizers included
- **Educational Examples**: Comprehensive examples and documentation

## 📋 Requirements

- Python 3.7+
- PyTorch 1.12.0+
- NumPy 1.21.0+
- Regex 2022.1.18+

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd pico-gpt

# Install dependencies
pip install -r requirements.txt
```

### 2. Test the Implementation

```bash
# Run basic functionality tests
python example.py
```

Expected output:
```
Pico-GPT Example Usage
==============================
=== Testing Forward Pass ===
Created tiny GPT with 106,496 parameters
Input shape: torch.Size([2, 10])
Output logits shape: torch.Size([2, 10, 100])
Loss: 4.6921
...
```

### 3. Train a Model

#### Option A: Quick Training (Recommended for testing)
```bash
# Train a small model quickly (5-10 minutes)
python train_small.py
```

#### Option B: Full Training
```bash
# Train with default settings (longer training)
python train.py
```

Training output:
```
Training small model for testing...
Training data: 5202 tokens
Model parameters: 620,672
iter   0: loss 5.2850
iter 100: loss 2.1402
iter 200: loss 1.2375
...
Model saved as 'pico_gpt_model.pt'
```

### 4. Generate Text

```bash
# Basic text generation
python generate.py --prompt "Hello world"

# Advanced generation with parameters
python generate.py --prompt "Python is" --max_tokens 50 --temperature 0.8 --top_k 10
```

Example output:
```
Using device: cpu
Model loaded from pico_gpt_model.pt

Prompt: Hello world
Generated: Hello world! This is a simple test. The quick brown fox jumps over the lazy dog.
```

## 📁 Project Structure

```
pico-gpt/
├── pico_gpt.py          # Core GPT model implementation
├── tokenizer.py         # Tokenization utilities
├── train.py             # Full training script
├── train_small.py       # Quick training for testing
├── generate.py          # Text generation CLI
├── example.py           # Educational examples
├── test_train.py        # Quick training test
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── TEST_RESULTS.md     # Test validation results
```

## 🔧 Configuration

### Model Architecture

Edit the `GPTConfig` class in `pico_gpt.py` or create custom configs:

```python
from pico_gpt import GPTConfig

# Small model (fast training)
config = GPTConfig()
config.block_size = 256      # Context length
config.vocab_size = 50304    # Vocabulary size
config.n_layer = 6          # Number of transformer layers
config.n_head = 6           # Number of attention heads
config.n_embd = 384         # Embedding dimension
config.dropout = 0.2        # Dropout rate

# Tiny model (very fast)
config.n_layer = 2
config.n_head = 2
config.n_embd = 64
```

### Training Parameters

Modify training scripts for different setups:

```python
# Training hyperparameters
batch_size = 16             # Batch size
learning_rate = 3e-4        # Learning rate
max_iters = 5000           # Training iterations
eval_interval = 1000       # Validation frequency
```

## 📚 Usage Examples

### Custom Training Data

```python
# train_custom.py
import torch
from pico_gpt import GPT, GPTConfig
from tokenizer import SimpleTokenizer

# Load your text data
with open('your_data.txt', 'r') as f:
    text = f.read()

# Create tokenizer and encode
tokenizer = SimpleTokenizer()
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# Train model (see train.py for full example)
```

### Programmatic Generation

```python
from pico_gpt import GPT
from generate import load_model, generate_text

# Load trained model
model, tokenizer = load_model('pico_gpt_model.pt')

# Generate text
result = generate_text(
    model, tokenizer, 
    prompt="The future of AI",
    max_tokens=100,
    temperature=0.7,
    top_k=15
)
print(result)
```

### Model Analysis

```python
from pico_gpt import GPT, GPTConfig

config = GPTConfig()
model = GPT(config)

print(f"Total parameters: {model.get_num_params():,}")
print(f"Non-embedding parameters: {model.get_num_params(non_embedding=True):,}")

# Analyze model components
for name, module in model.named_modules():
    if hasattr(module, 'weight'):
        print(f"{name}: {module.weight.shape}")
```

## 🎯 Command Line Interface

### generate.py Options

```bash
python generate.py --help
```

Available options:
- `--model`: Path to model checkpoint (default: `pico_gpt_model.pt`)
- `--prompt`: Text prompt for generation (default: `"Hello"`)
- `--max_tokens`: Maximum tokens to generate (default: `100`)
- `--temperature`: Sampling temperature 0.1-2.0 (default: `0.8`)
- `--top_k`: Top-k sampling parameter (default: `10`)

### Examples

```bash
# Creative writing (high temperature)
python generate.py --prompt "Once upon a time" --temperature 1.2 --max_tokens 200

# Factual completion (low temperature)
python generate.py --prompt "Python is a programming language" --temperature 0.3

# Diverse sampling
python generate.py --prompt "The benefits of" --top_k 20 --temperature 0.9
```

## 🧪 Testing & Validation

Run the test suite to verify everything works:

```bash
# Quick functionality test
python example.py

# Training validation
python test_train.py

# Full integration test
python train_small.py && python generate.py --prompt "Test"
```

See `TEST_RESULTS.md` for detailed validation results.

## 🎓 Educational Notes

### Understanding the Architecture

1. **Transformer Blocks**: Each block contains self-attention + MLP
2. **Causal Attention**: Prevents looking at future tokens
3. **Position Embeddings**: Help model understand token order
4. **Layer Normalization**: Stabilizes training
5. **Weight Tying**: Input/output embeddings share weights

### Key Components

- `CausalSelfAttention`: Implements masked multi-head attention
- `MLP`: Feed-forward network with GELU activation
- `Block`: Complete transformer block
- `GPT`: Full model with embeddings and language modeling head

### Training Process

1. **Tokenization**: Convert text to integer sequences
2. **Batching**: Create input/target pairs
3. **Forward Pass**: Compute predictions and loss
4. **Backpropagation**: Update model weights
5. **Generation**: Sample from learned distribution

## ⚡ Performance Tips

### For Faster Training
- Reduce `n_layer`, `n_head`, `n_embd`
- Use smaller `block_size` and `batch_size`
- Enable GPU if available: `device = 'cuda'`

### For Better Generation
- Train longer (more iterations)
- Use larger vocabulary
- Increase model size
- Tune temperature and top_k

### Memory Optimization
- Reduce batch size if out of memory
- Use gradient checkpointing for large models
- Consider mixed precision training

## 🐛 Troubleshooting

### Common Issues

**"CUDA out of memory"**
```bash
# Reduce batch size or model size
config.n_embd = 256  # instead of 512
batch_size = 8       # instead of 16
```

**"Model file not found"**
```bash
# Train a model first
python train_small.py
```

**Poor generation quality**
```bash
# Train longer or with more data
max_iters = 10000    # increase training steps
```

**Slow training**
```bash
# Use smaller model for testing
config.n_layer = 2
config.n_embd = 64
```

## 📈 Next Steps

- **Experiment** with different model sizes and hyperparameters
- **Add your own data** for domain-specific models  
- **Implement** additional features like beam search
- **Scale up** to larger models and datasets
- **Compare** with other transformer implementations

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

---

*Happy training! 🚀*
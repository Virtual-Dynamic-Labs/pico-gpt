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

#### Option A: Simple Generation
```bash
# Basic text generation
python generate.py --prompt "Hello world"

# Advanced generation with parameters
python generate.py --prompt "Python is" --max_tokens 50 --temperature 0.8 --top_k 10
```

#### Option B: Interactive Chat (Recommended)
```bash
# Start interactive conversation mode
python cli_client.py

# Or use PowerShell launcher on Windows
.\run_cli.ps1

# Single prompt mode
python cli_client.py --prompt "Tell me about AI"
```

**Interactive CLI Features:**
- 💬 **Conversation Mode** - Maintains context across exchanges
- 🔄 **Single-Prompt Mode** - Independent text generation
- ⚙️ **Adjustable Settings** - Temperature, top-k, max tokens
- 📝 **Command System** - `/help`, `/settings`, `/clear`, etc.
- 💾 **History Support** - Command history with readline

Example CLI session:
```
*** Pico GPT CLI Client - Conversation Mode ***
[You]: Hello, my name is Alice.
[Assistant]: Nice to meet you, Alice! How can I help you today?

[You]: What's my name?
[Assistant]: Your name is Alice, as you mentioned when we started talking.

[You]: /settings
[SETTINGS] Current Generation Settings:
  Max tokens: 100
  Temperature: 0.8
  Top-k: 20
```

## 📁 Project Structure

```
pico-gpt/
├── pico_gpt.py              # Core GPT model implementation
├── tokenizer.py             # Tokenization utilities
├── train.py                 # Full training script
├── train_large.py           # Large model training script
├── train_small.py           # Quick training for testing
├── generate.py              # Text generation CLI
├── cli_client.py            # Interactive conversation CLI
├── run_cli.ps1              # PowerShell launcher for Windows
├── example.py               # Educational examples
├── test_train.py            # Quick training test
├── requirements.txt         # Python dependencies
├── benchmarks/              # Performance benchmarking tools
│   ├── benchmark_cuda_vs_cpu.py
│   ├── benchmark_large_model.py
│   └── test_large_model.py
├── data/                    # Training datasets
│   ├── combined_literature.txt
│   └── [various text files...]
├── *.pt                     # Model checkpoint files (see below)
├── README.md               # This file
└── TEST_RESULTS.md         # Test validation results
```

## 💾 Model Files (.pt files)

The project contains several PyTorch model checkpoint files:

### **Active Models**
- **`pico_gpt_large_best.pt`** (299MB) - **Primary model** with best validation performance
  - 25.7M parameters (8 layers, 8 heads, 512 embedding dim)
  - Validation loss: 1.1405
  - Trained for 4,801 iterations
  - **Default model** used by CLI

- **`pico_gpt_model.pt`** (2.5MB) - Compact model for testing
  - 620K parameters (smaller, faster alternative)
  - Good for quick testing and low-resource environments

### **Training Checkpoints**
- `pico_gpt_large_checkpoint_*.pt` - Saved every 500 iterations during training
- `pico_gpt_large_final.pt` - Final training state after 5,000 iterations

### **What's Inside Each .pt File**
```python
checkpoint = torch.load('model.pt')
# Contains:
checkpoint['model_state_dict']    # Neural network weights
checkpoint['config']              # Model architecture settings
checkpoint['tokenizer']           # Text encoding/decoding component
checkpoint['iter']                # Training iteration count
checkpoint['best_val_loss']       # Best validation loss achieved
```

### **File Size Reference**
- **Large models**: ~299MB each (25.7M parameters)
- **Small model**: ~2.5MB (620K parameters)
- **Total checkpoint storage**: ~3GB (can be cleaned up if needed)

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

### Interactive CLI (cli_client.py)

The main interface for interacting with trained models:

```bash
python cli_client.py --help
```

**Options:**
- `--model`, `-m`: Model file path (default: `pico_gpt_large_best.pt`)
- `--device`, `-d`: Device (`cpu`, `cuda`, `auto`)
- `--max-tokens`, `-t`: Max tokens to generate (default: `100`)
- `--temperature`, `-T`: Sampling temperature (default: `0.8`)
- `--top-k`, `-k`: Top-k sampling (default: `20`)
- `--prompt`, `-p`: Single prompt mode (non-interactive)

**Interactive Commands:**
- `/help` - Show command reference
- `/settings` - View/modify generation parameters
- `/clear` - Clear screen
- `/reset` - Clear conversation context
- `/status` - Show conversation status
- `/mode` - Toggle conversation/single-prompt modes
- `/quit` - Exit program
- `/info` - Show model information

### Simple Generation (generate.py)

```bash
python generate.py --help
```

Available options:
- `--model`: Path to model checkpoint (default: `pico_gpt_large_best.pt`)
- `--prompt`: Text prompt for generation (default: `"Hello"`)
- `--max_tokens`: Maximum tokens to generate (default: `100`)
- `--temperature`: Sampling temperature 0.1-2.0 (default: `0.8`)
- `--top_k`: Top-k sampling parameter (default: `10`)

### Examples

```bash
# Interactive conversation
python cli_client.py

# Windows PowerShell
.\run_cli.ps1 -Prompt "Hello world"

# Creative writing (high temperature)
python generate.py --prompt "Once upon a time" --temperature 1.2 --max_tokens 200

# Factual completion (low temperature)
python generate.py --prompt "Python is a programming language" --temperature 0.3

# Different model
python cli_client.py --model pico_gpt_model.pt --prompt "Test small model"
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

## ⚡ Performance Benchmarking

### CUDA vs CPU Performance
```bash
# Benchmark inference performance
python benchmarks/benchmark_large_model.py

# Training performance comparison (small model)
python benchmarks/benchmark_cuda_vs_cpu.py
```

**Example Results (RTX 3080 Ti):**
```
Large Model CUDA vs CPU Inference Benchmark
============================================================
Model: 123.7M parameters (12 layers, 12 heads, 768 embedding dim)

Average Generation Time:
  CPU:  2739.40 ms
  CUDA: 430.50 ms
  Speedup: 6.36x faster

Throughput (tokens/sec):
  CPU:  18.3
  CUDA: 116.1
  Improvement: 6.36x

Memory Usage:
  CUDA Allocated: 981.1 MB
```

### Model Comparison
| Model | Parameters | Size | Inference Speed (CUDA) | Use Case |
|-------|------------|------|----------------------|----------|
| `pico_gpt_model.pt` | 620K | 2.5MB | ~500 tokens/sec | Testing, low-resource |
| `pico_gpt_large_best.pt` | 25.7M | 299MB | ~116 tokens/sec | Production, quality |

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
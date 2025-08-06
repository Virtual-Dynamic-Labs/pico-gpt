# 🤖 Pico-GPT

A minimal, educational implementation of the GPT (Generative Pre-trained Transformer) architecture in PyTorch with clean, professional project structure. This project provides a complete, well-documented codebase for understanding how GPT models work under the hood.

## 🌟 Features

- **Complete GPT Architecture**: Multi-head self-attention, position embeddings, layer normalization
- **Professional Structure**: Organized by purpose with src/, training/, cli/, tests/
- **Multiple Interfaces**: Interactive CLI, direct generation, Python API
- **Optimized Training**: Ultra-fast conversation training (16 seconds)
- **GPU Acceleration**: Automatic CUDA detection and usage
- **Configurable Models**: Easily adjust layers, heads, embedding dimensions
- **Text Generation**: Autoregressive generation with temperature and top-k sampling
- **Educational Examples**: Comprehensive examples and documentation

## 📋 Requirements

- Python 3.7+
- PyTorch 1.12.0+
- NumPy 1.21.0+
- Regex 2022.1.18+

## 📁 Project Structure

```
pico-gpt/
├── 📁 src/                      # Core implementation
│   ├── pico_gpt.py             # Main GPT model & architecture
│   ├── tokenizer.py            # Simple & BPE tokenizers
│   ├── fast_tokenizer.py       # Optimized GPT-2 style tokenizer
│   └── __init__.py
│
├── 📁 training/                 # Training scripts
│   ├── train_final.py          # 🌟 BEST: Fast conversation training
│   ├── train_fast.py           # Speed-optimized training
│   ├── train_conversation.py   # Conversation-focused training
│   ├── train_large.py          # Large model training
│   ├── train_small.py          # Quick testing
│   └── train.py                # Basic training
│
├── 📁 cli/                      # User interfaces
│   ├── cli_fast.py             # 🌟 MAIN: Interactive chat CLI
│   ├── cli_client.py           # Alternative CLI
│   ├── generate.py             # Simple text generation
│   └── run_cli.ps1             # Windows PowerShell launcher
│
├── 📁 models/                   # Trained models
│   ├── pico_gpt_final.pt       # 🌟 BEST: Fast conversation model
│   ├── pico_gpt_fast.pt        # Speed-optimized model
│   ├── pico_gpt_large_best.pt  # Large production model (25.7M params)
│   └── pico_gpt_model.pt       # Compact testing model (620K params)
│
├── 📁 datasets/                 # Training data & tokenizers
│   ├── clean_conversation_data.txt    # 🌟 Clean chat data
│   ├── fast_tokenizer_gpt2_8000.pkl  # 🌟 Optimized tokenizer
│   └── [other datasets...]
│
├── 📁 data/                     # Raw training data
│   ├── combined_literature.txt  # Literature corpus
│   └── [classic books...]
│
├── 📁 tests/                    # Test & example scripts
│   ├── example.py              # Basic functionality demo
│   ├── test_conversation.py    # Conversation testing
│   └── debug_conversation.py   # Debugging tools
│
├── 📁 scripts/                  # Utility scripts
├── 📁 benchmarks/               # Performance testing
├── 📄 main.py                   # 🌟 Main entry point
├── 📄 run.py                    # Simple runner
├── 📄 architecture_diagram.md   # Visual architecture guide
└── 📄 README.md                 # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd pico-gpt

# Install dependencies
pip install -r requirements.txt
```

### 2. Interactive Chat (Recommended)

```bash
# Using the simple runner
python run.py

# Using main entry point
python main.py chat

# Windows batch file
run_cli.bat

# Windows PowerShell
.\run_cli.ps1
```

**Interactive CLI Features:**
- 💬 **Conversation Mode** - Maintains context across exchanges
- 🔄 **Single-Prompt Mode** - Independent text generation
- ⚙️ **Adjustable Settings** - Temperature, top-k, max tokens
- 📝 **Command System** - `/help`, `/settings`, `/clear`, etc.
- 💾 **History Support** - Command history with readline

### 3. Train a Model

```bash
# Train the best conversation model (16 seconds!)
cd training
python train_final.py

# Or from root
python main.py train --type conversation

# Quick testing model
python training/train_small.py

# Large production model
python training/train_large.py
```

### 4. Generate Text

```bash
# Simple generation
python cli/generate.py --prompt "Hello world"

# Advanced generation with parameters
python cli/generate.py --prompt "Python is" --max_tokens 50 --temperature 0.8 --top_k 10

# From main entry point
python main.py generate --prompt "Once upon a time"
```

### 5. Test the Implementation

```bash
# Basic functionality test
python tests/example.py

# Training test
python tests/test_train.py

# From root
python main.py test --type basic
```

## 💾 Model Files

### **Active Models**
- **`pico_gpt_large_best.pt`** (299MB) - **Primary model** with best validation performance
  - 25.7M parameters (8 layers, 8 heads, 512 embedding dim)
  - Validation loss: 1.1405
  - **Default model** used by CLI

- **`pico_gpt_final.pt`** - **Fast conversation model**
  - 13.7M parameters, trains in 16 seconds
  - Conversation-focused, no literature regurgitation

- **`pico_gpt_model.pt`** (2.5MB) - Compact model for testing
  - 620K parameters (smaller, faster alternative)
  - Good for quick testing and low-resource environments

### **Model Comparison**
| Model | Parameters | Size | Training Time | Use Case |
|-------|------------|------|---------------|----------|
| `pico_gpt_final.pt` | 13.7M | ~50MB | 16 seconds | 🌟 **Best for chat** |
| `pico_gpt_model.pt` | 620K | 2.5MB | ~2 minutes | Testing, low-resource |
| `pico_gpt_large_best.pt` | 25.7M | 299MB | ~2 hours | Production, quality |

## 🎯 Command Line Interface

### Interactive CLI Commands

When in conversation mode, you can use these commands:

- `/help` - Show command reference
- `/settings` - View/modify generation parameters
- `/clear` - Clear screen
- `/reset` - Clear conversation context
- `/status` - Show conversation status
- `/mode` - Toggle conversation/single-prompt modes
- `/info` - Show model information
- `/load` - Load a different model
- `/quit` - Exit program

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` / `-m` | Path to model file | `pico_gpt_large_best.pt` |
| `--device` / `-d` | Device (cpu/cuda/auto) | `auto` |
| `--max-tokens` / `-t` | Maximum tokens to generate | `100` |
| `--temperature` / `-T` | Sampling temperature | `0.8` |
| `--top-k` / `-k` | Top-k sampling | `20` |
| `--prompt` / `-p` | Single prompt mode | None |

### Examples

```bash
# Interactive conversation
python cli/cli_client.py

# Windows PowerShell with custom settings
.\cli\run_cli.ps1 -Model "pico_gpt_final.pt" -MaxTokens 200 -Temperature 0.9

# Creative writing (high temperature)
python cli/generate.py --prompt "Once upon a time" --temperature 1.2 --max_tokens 200

# Factual completion (low temperature)
python cli/generate.py --prompt "Python is a programming language" --temperature 0.3

# Different model
python cli/cli_client.py --model models/pico_gpt_final.pt --prompt "Test conversation"
```

## 🔧 Configuration

### Model Architecture

Edit the `GPTConfig` class in `src/pico_gpt.py` or create custom configs:

```python
from src.pico_gpt import GPTConfig

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

### Python API

```python
from src.pico_gpt import GPT, GPTConfig
from src.fast_tokenizer import GPT2LikeTokenizer
import torch

# Load model
checkpoint = torch.load('models/pico_gpt_final.pt')
model = GPT(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
tokenizer = checkpoint['tokenizer']

# Generate text
model.eval()
context = torch.tensor(tokenizer.encode("Hello"), dtype=torch.long).unsqueeze(0)
generated = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=10)
result = tokenizer.decode(generated[0].tolist())
print(result)
```

### Custom Training Data

```python
# training/train_custom.py
import torch
from src.pico_gpt import GPT, GPTConfig
from src.tokenizer import SimpleTokenizer

# Load your text data
with open('your_data.txt', 'r') as f:
    text = f.read()

# Create tokenizer and encode
tokenizer = SimpleTokenizer()
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# Train model (see training scripts for full examples)
```

## ⚡ Performance Benchmarking

### CUDA vs CPU Performance

```bash
# Benchmark inference performance
python benchmarks/benchmark_large_model.py

# Training performance comparison
python benchmarks/benchmark_cuda_vs_cpu.py
```

**Example Results (RTX 3080 Ti):**
```
Large Model CUDA vs CPU Inference Benchmark
============================================================
Model: 25.7M parameters (8 layers, 8 heads, 512 embedding dim)

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

## 🎓 Educational Notes

### Architecture Diagram

See [architecture_diagram.md](architecture_diagram.md) for a detailed visual representation of the Pico GPT architecture, including data flow and component relationships.

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
# Use CPU or reduce batch size
python cli/cli_client.py --device cpu
# Or reduce model size in config
```

**"Model file not found"**
```bash
# Train a model first
python training/train_small.py
```

**Poor generation quality**
```bash
# Train longer or with more data
# Use train_final.py for best conversation quality
```

**PowerShell Execution Policy (Windows)**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## ✅ Test Results

All core functionality has been verified:

### Verified Features
✅ **Transformer Architecture** - Multi-head self-attention with causal masking  
✅ **Training Pipeline** - Proper data batching, AdamW optimizer, checkpointing  
✅ **Text Generation** - Autoregressive generation with temperature/top-k sampling  
✅ **Tokenization** - Character-level and BPE tokenizers  
✅ **CLI Interface** - Interactive conversation and single-prompt modes  
✅ **GPU Support** - CUDA acceleration with automatic detection  

### Performance Characteristics
- **Training Speed**: 16 seconds for conversation model, ~143 tokens/second on CPU
- **Memory Usage**: Efficient for both small and large models
- **Convergence**: Good learning curves observed
- **Generation Quality**: Coherent outputs for trained models

## 📈 What's New (Post-Refactor)

### **Before Refactoring** ❌
- Everything scattered in root folder
- Training scripts mixed with core code
- Hard to navigate and maintain
- Import path chaos

### **After Refactoring** ✅
- **Clean structure**: Logical folder organization
- **Separated concerns**: Each folder has one purpose
- **Professional**: Industry-standard project layout
- **Maintainable**: Easy to find and modify code
- **Modular**: Components can be imported independently

## 📝 Integration Examples

### Batch Scripts (Windows)
```batch
@echo off
python cli/cli_client.py --prompt "%1" --max-tokens 100 > output.txt
echo Generated text saved to output.txt
```

### PowerShell Functions
```powershell
function Generate-Text {
    param([string]$Prompt)
    python cli/cli_client.py --prompt $Prompt --max-tokens 100
}
```

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

---

*Happy training! 🚀*

**Ready for:**
- Educational purposes and learning transformers
- Experimentation with GPT architectures  
- Small-scale language modeling tasks
- Research and development
- Production use with proper scaling
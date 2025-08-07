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
│   └── train_conversation.py   # 🌟 BEST: Conversation model training
│
├── 📁 cli/                      # User interfaces
│   ├── cli_client.py           # 🌟 MAIN: Interactive chat CLI
│   └── generate.py             # Simple text generation
│
├── 📁 models/                   # Trained models
│   └── pico_gpt_conversation.pt # 🌟 Conversation model (26.2M params)
│
├── 📁 datasets/                 # Training data & tokenizers
│   ├── clean_conversation_data.txt      # 🌟 Clean chat data
│   ├── fast_tokenizer_gpt2_8000.pkl    # 🌟 Optimized tokenizer
│   ├── combined_enhanced_data.txt       # Enhanced training data
│   ├── comprehensive_conversations.txt  # Comprehensive dialogue data
│   ├── conversation_training.txt        # Core training conversations
│   ├── smart_reasoning_data.txt         # Advanced reasoning examples
│   └── [other datasets...]
│
├── 📁 tests/                    # Test & example scripts
│   ├── example.py              # Basic functionality demo
│   ├── test_conversation.py    # Conversation testing
│   ├── debug_conversation.py   # Debugging tools
│   └── test_train.py           # Training verification
│
├── 📁 scripts/                  # Utility scripts
│   ├── create_clean_conversation_data.py  # Data preprocessing
│   ├── create_conversation_data.py        # Conversation generation
│   ├── create_smart_dataset.py           # Smart dataset creation
│   ├── download_dataset.py               # Dataset downloading
│   ├── main.py                           # Main entry point
│   └── run.py                            # Simple runner
│
├── 📁 benchmarks/               # Performance testing
│   ├── benchmark_cuda_vs_cpu.py      # CUDA vs CPU benchmarking
│   ├── benchmark_large_model.py      # Large model performance
│   └── test_large_model.py           # Large model testing
│
├── 📄 setup.py                  # Package configuration
├── 📄 requirements.txt          # Python dependencies
├── 📄 upload_to_hf.py           # Hugging Face upload script
├── 📄 run_cli.ps1               # Windows PowerShell launcher
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
# Using main scripts
python scripts/run.py
python scripts/main.py

# Using CLI directly
python cli/cli_client.py

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
# Train the conversation model
python training/train_conversation.py
```

### 4. Generate Text

```bash
# Simple generation
python cli/generate.py --prompt "Hello world"

# Advanced generation with parameters
python cli/generate.py --prompt "Python is" --max_tokens 50 --temperature 0.8 --top_k 10

# From scripts directory
python scripts/main.py generate --prompt "Once upon a time"
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
- **`pico_gpt_conversation.pt`** - **Primary conversation model**
  - 26.2M parameters (8 layers, 8 heads, 512 embedding dim)
  - **Default model** used by CLI
  - Optimized for natural conversations

- **`pico_gpt_large.pt`** - **Large capability model**
  - 88.9M parameters (12 layers, 12 heads, 768 embedding dim)
  - Maximum model capability
  - Use for complex tasks requiring more intelligence

### **Model Comparison**
| Model | Parameters | Size | Use Case |
|-------|------------|------|---------|
| `pico_gpt_conversation.pt` | 26.2M | ~100MB | 🌟 **Best for conversation** |
| `pico_gpt_large.pt` | 88.9M | ~350MB | Maximum capability, complex tasks |

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
| `--model` / `-m` | Path to model file | `pico_gpt_conversation.pt` |
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
.\cli\run_cli.ps1 -Model "pico_gpt_large.pt" -MaxTokens 200 -Temperature 0.9

# Creative writing (high temperature)
python cli/generate.py --prompt "Once upon a time" --temperature 1.2 --max_tokens 200

# Factual completion (low temperature)
python cli/generate.py --prompt "Python is a programming language" --temperature 0.3

# Different model
python cli/cli_client.py --model models/pico_gpt_conversation.pt --prompt "Test conversation"
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
checkpoint = torch.load('models/pico_gpt_conversation.pt', weights_only=False)
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

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              PICO GPT ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                                INPUT LAYER                              │
│                                                                         │
│  Input Text: "Hello world"                                              │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────┐                                                    │
│  │   TOKENIZER     │  Character/BPE tokenization                       │
│  │  SimpleTokenizer│  "Hello world" → [72, 101, 108, 108, 111, ...]    │
│  │  BPETokenizer   │                                                    │
│  └─────────────────┘                                                    │
│       │                                                                 │
│       ▼                                                                 │
│  Token IDs: [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]     │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           EMBEDDING LAYERS                              │
│                                                                         │
│  ┌──────────────────┐    ┌──────────────────┐                          │
│  │ Token Embedding  │    │Position Embedding│                          │
│  │    (wte)         │    │     (wpe)        │                          │
│  │  vocab_size      │    │   block_size     │                          │
│  │     ↓            │    │      ↓           │                          │
│  │  n_embd dim      │    │   n_embd dim     │                          │
│  └──────────────────┘    └──────────────────┘                          │
│            │                       │                                   │
│            └───────────┬───────────┘                                   │
│                        ▼                                               │
│                 ┌─────────────┐                                        │
│                 │  Element +  │                                        │
│                 │   Dropout   │                                        │
│                 └─────────────┘                                        │
│                        │                                               │
│                        ▼                                               │
│              Embedded Sequence                                         │
│             [batch_size, seq_len, n_embd]                             │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRANSFORMER BLOCKS (n_layer)                    │
│                                                                         │
│  ┌─ BLOCK 1 ──────────────────────────────────────────────────────────┐ │
│  │                                                                    │ │
│  │  Input: x                                                          │ │
│  │     │                                                              │ │
│  │     ▼                                                              │ │
│  │  ┌──────────────┐                                                  │ │
│  │  │ Layer Norm 1 │                                                  │ │
│  │  └──────────────┘                                                  │ │
│  │     │                                                              │ │
│  │     ▼                                                              │ │
│  │  ┌──────────────────────────────────────────────┐                  │ │
│  │  │         CAUSAL SELF-ATTENTION               │                  │ │
│  │  │                                             │                  │ │
│  │  │  ┌─────────┐                                │                  │ │
│  │  │  │ Q,K,V   │  Linear projection             │                  │ │
│  │  │  │ Linear  │  (n_embd → 3 * n_embd)         │                  │ │
│  │  │  └─────────┘                                │                  │ │
│  │  │      │                                      │                  │ │
│  │  │      ▼                                      │                  │ │
│  │  │  ┌─────────┐                                │                  │ │
│  │  │  │Multi-Head│  Split into n_head            │                  │ │
│  │  │  │Attention │  Compute attention weights    │                  │ │
│  │  │  │         │  Apply causal mask             │                  │ │
│  │  │  └─────────┘                                │                  │ │
│  │  │      │                                      │                  │ │
│  │  │      ▼                                      │                  │ │
│  │  │  ┌─────────┐                                │                  │ │
│  │  │  │Output   │  Concatenate heads             │                  │ │
│  │  │  │Linear   │  Project back (n_embd)         │                  │ │
│  │  │  │+Dropout │                                │                  │ │
│  │  │  └─────────┘                                │                  │ │
│  │  └──────────────────────────────────────────────┘                  │ │
│  │     │                                                              │ │
│  │     ▼                                                              │ │
│  │  ┌──────────┐    ◄── Residual Connection                           │ │
│  │  │    +     │                                                      │ │
│  │  └──────────┘                                                      │ │
│  │     │                                                              │ │
│  │     ▼                                                              │ │
│  │  ┌──────────────┐                                                  │ │
│  │  │ Layer Norm 2 │                                                  │ │
│  │  └──────────────┘                                                  │ │
│  │     │                                                              │ │
│  │     ▼                                                              │ │
│  │  ┌──────────────────────────────────────────────┐                  │ │
│  │  │                 MLP                          │                  │ │
│  │  │                                              │                  │ │
│  │  │  ┌─────────────┐                            │                  │ │
│  │  │  │   Linear    │  n_embd → 4 * n_embd       │                  │ │
│  │  │  │  (c_fc)     │                            │                  │ │
│  │  │  └─────────────┘                            │                  │ │
│  │  │         │                                   │                  │ │
│  │  │         ▼                                   │                  │ │
│  │  │  ┌─────────────┐                            │                  │ │
│  │  │  │    GELU     │  Activation function       │                  │ │
│  │  │  └─────────────┘                            │                  │ │
│  │  │         │                                   │                  │ │
│  │  │         ▼                                   │                  │ │
│  │  │  ┌─────────────┐                            │                  │ │
│  │  │  │   Linear    │  4 * n_embd → n_embd       │                  │ │
│  │  │  │ (c_proj)    │                            │                  │ │
│  │  │  │  +Dropout   │                            │                  │ │
│  │  │  └─────────────┘                            │                  │ │
│  │  └──────────────────────────────────────────────┘                  │ │
│  │     │                                                              │ │
│  │     ▼                                                              │ │
│  │  ┌──────────┐    ◄── Residual Connection                           │ │
│  │  │    +     │                                                      │ │
│  │  └──────────┘                                                      │ │
│  │     │                                                              │ │
│  │     ▼  Output to next block                                        │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─ BLOCK 2...n ────────────────────────────────────────────────────┐   │
│  │                    (Same structure)                              │   │
│  └───────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           FINAL OUTPUT LAYER                           │
│                                                                         │
│  ┌──────────────────┐                                                   │
│  │ Final Layer Norm │  Normalize final transformer output              │
│  │     (ln_f)       │                                                  │
│  └──────────────────┘                                                   │
│           │                                                            │
│           ▼                                                            │
│  ┌──────────────────┐                                                   │
│  │ Language Model   │  Linear: n_embd → vocab_size                     │
│  │ Head (lm_head)   │  Weight tied with input embeddings              │
│  └──────────────────┘                                                   │
│           │                                                            │
│           ▼                                                            │
│     Logits: [batch_size, seq_len, vocab_size]                         │
│                                                                         │
│  For Training:                    For Generation:                      │
│  ┌────────────────┐               ┌─────────────────┐                  │
│  │ Cross Entropy  │               │  Temperature    │                  │
│  │     Loss       │               │    Scaling      │                  │
│  │                │               │       │         │                  │
│  │ Compare with   │               │       ▼         │                  │
│  │ target tokens  │               │  ┌──────────┐   │                  │
│  └────────────────┘               │  │ Top-k    │   │                  │
│                                   │  │Sampling  │   │                  │
│                                   │  └──────────┘   │                  │
│                                   │       │         │                  │
│                                   │       ▼         │                  │
│                                   │  ┌──────────┐   │                  │
│                                   │  │Multinomial   │                  │
│                                   │  │ Sampling │   │                  │
│                                   │  └──────────┘   │                  │
│                                   └─────────────────┘                  │
│                                           │                            │
│                                           ▼                            │
│                                  Next Token Prediction                 │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                            KEY COMPONENTS                               │
│                                                                         │
│ Configuration (GPTConfig):                                             │
│  • block_size: 1024 (max sequence length)                             │
│  • vocab_size: 50304 (vocabulary size)                                │
│  • n_layer: 12 (number of transformer blocks)                         │
│  • n_head: 12 (number of attention heads)                             │
│  • n_embd: 768 (embedding dimension)                                  │
│  • dropout: 0.0 (dropout rate)                                        │
│                                                                         │
│ Training Components:                                                    │
│  • Training Loop: batch processing, loss calculation                   │
│  • AdamW Optimizer: weight decay regularization                        │
│  • Learning Rate Scheduling: configurable learning rate               │
│  • Validation: periodic loss evaluation                                │
│  • Checkpointing: model state saving                                   │
│                                                                         │
│ Generation Features:                                                    │
│  • Autoregressive: generates one token at a time                       │
│  • Temperature Sampling: controls randomness                           │
│  • Top-k Sampling: limits vocabulary for coherent output               │
│  • Causal Masking: prevents looking ahead during attention             │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                            DATA FLOW SUMMARY                            │
│                                                                         │
│  Text Input → Tokenization → Embeddings → Transformer Blocks →        │
│  → Layer Norm → Language Model Head → Logits → Sampling → Output      │
│                                                                         │
│  Training: Input/Target pairs, Cross-entropy loss, Backpropagation     │
│  Generation: Autoregressive sampling with temperature and top-k        │
└─────────────────────────────────────────────────────────────────────────┘
```

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
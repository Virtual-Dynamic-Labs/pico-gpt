# 🤖 Pico-GPT - Refactored & Organized

A minimal, educational implementation of GPT (Generative Pre-trained Transformer) with **clean, professional project structure**.

## ✅ Refactored Project Structure

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
│   ├── pico_gpt_conversation.pt # Pure conversation model
│   └── [other model checkpoints...]
│
├── 📁 datasets/                 # Training data & tokenizers
│   ├── clean_conversation_data.txt    # 🌟 Clean chat data
│   ├── conversation_data.txt          # Extended chat data
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
│   ├── create_clean_conversation_data.py  # Generate chat data
│   └── create_conversation_data.py        # Generate extended data
│
├── 📁 benchmarks/               # Performance testing
│   ├── benchmark_large_model.py # CUDA vs CPU benchmarks
│   └── benchmark_cuda_vs_cpu.py # Training speed tests
│
├── 📄 main.py                   # 🌟 Main entry point
├── 📄 run.py                    # Simple runner
├── 📄 setup.py                  # Package installation
├── 📄 requirements.txt          # Dependencies
└── 📄 README.md                 # This file
```

## 🚀 Quick Start (Refactored)

### 1. **Chat with the Model** (Recommended)
```bash
# Using the simple runner
python run.py

# Or using the main entry point
python main.py chat

# Test with a specific prompt
python run.py --prompt "Hello, how are you?"
```

### 2. **Train a New Model**
```bash
# Train the best conversation model (16 seconds!)
cd training
python train_final.py

# Or from root
python main.py train --type conversation
```

### 3. **Generate Text**
```bash
cd cli
python cli_fast.py --prompt "Once upon a time"

# Or from root
python main.py generate --prompt "Hello world"
```

### 4. **Run Tests**
```bash
cd tests
python example.py

# Or from root  
python main.py test --type basic
```

## 🌟 Key Features After Refactoring

### **Optimized Models**
- **`models/pico_gpt_final.pt`** - 13.7M params, 16s training, conversation-focused
- **Ultra-fast training**: 23x faster than before
- **Proper conversation data**: No more literature regurgitation
- **Efficient tokenizer**: GPT-2 style with 4.27x compression

### **Professional Structure**
- ✅ **Organized by purpose**: src/, training/, cli/, tests/
- ✅ **Clean imports**: Proper Python package structure
- ✅ **Separated concerns**: Data, models, training, interfaces
- ✅ **Easy navigation**: No more scattered files

### **Multiple Interfaces**
- **CLI**: Interactive conversation mode
- **API**: Direct Python import from src/
- **Scripts**: Utility functions in scripts/
- **Main**: Unified entry point with subcommands

## 📊 Performance (Post-Refactor)

| Model | Size | Training Time | Use Case |
|-------|------|---------------|----------|
| `pico_gpt_final.pt` | 13.7M | 16 seconds | 🌟 **Best for chat** |
| `pico_gpt_fast.pt` | 13.7M | 72 seconds | Speed testing |
| `pico_gpt_conversation.pt` | 29.3M | 4 minutes | Extended conversations |

**CUDA Performance**: 2.5x speedup, 114MB memory usage

## 🔧 Installation & Setup

```bash
# Clone and install
git clone <repo-url>
cd pico-gpt

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .

# Quick test
python run.py --prompt "Hello!"
```

## 🎯 What's Fixed

### **Before Refactoring** ❌
- Everything scattered in root folder
- Training scripts mixed with core code
- Test files everywhere
- Import path chaos
- Hard to navigate

### **After Refactoring** ✅
- **Clean structure**: Logical folder organization
- **Separated concerns**: Each folder has one purpose
- **Easy imports**: Proper Python package structure
- **Professional**: Industry-standard project layout
- **Maintainable**: Easy to find and modify code

## 📚 Usage Examples

### **Quick Chat**
```bash
python run.py
# Starts interactive conversation mode
```

### **Training**
```bash
cd training
python train_final.py
# Trains conversation model in 16 seconds
```

### **Benchmarking**
```bash
cd benchmarks
python benchmark_large_model.py
# Tests CUDA vs CPU performance
```

### **Python API**
```python
from src.pico_gpt import GPT, GPTConfig
from src.fast_tokenizer import GPT2LikeTokenizer

# Load model
import torch
checkpoint = torch.load('models/pico_gpt_final.pt')
model = GPT(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
```

## 🎉 Benefits of Refactoring

1. **🗂️ Organization**: Everything has its place
2. **🔍 Findability**: Easy to locate specific functionality  
3. **🚀 Maintainability**: Changes are isolated and clear
4. **📦 Modularity**: Components can be imported independently
5. **🏗️ Professionalism**: Industry-standard project structure
6. **🧪 Testability**: Tests are separated and organized
7. **📈 Scalability**: Easy to add new features

The codebase is now **production-ready** with proper organization!
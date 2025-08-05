# Pico GPT CLI Client

A command-line interface for interacting with your trained Pico GPT models on Windows.

## Features

- **🎯 Conversation Mode**: Long-running chat sessions with context memory
- **📝 Single Prompt Mode**: Generate text from individual prompts
- **🧠 Context Management**: Automatically manages conversation history
- **⚙️ Customizable Generation**: Adjust temperature, top-k, and max tokens
- **🖥️ Windows Optimized**: Native Windows batch and PowerShell launchers
- **💾 Command History**: Readline support with persistent history
- **🚀 GPU Support**: Automatic CUDA detection and usage

## Quick Start

### 1. Conversation Mode (Recommended)

```cmd
# Using batch file (simple)
run_cli.bat

# Using PowerShell (advanced)
.\run_cli.ps1
```

This starts an interactive conversation where the AI remembers context from your previous messages, creating a natural chat experience.

### 2. Single Prompt Mode

```cmd
# Batch file
run_cli.bat --prompt "Once upon a time"

# PowerShell
.\run_cli.ps1 -Prompt "Once upon a time"

# Direct Python
python cli_client.py --prompt "Once upon a time"
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` / `-m` | Path to model file | `pico_gpt_large_best.pt` |
| `--device` / `-d` | Device to use (cpu/cuda/auto) | `auto` |
| `--max-tokens` / `-t` | Maximum tokens to generate | `100` |
| `--temperature` / `-T` | Temperature for generation | `0.8` |
| `--top-k` / `-k` | Top-k sampling | `20` |
| `--prompt` / `-p` | Single prompt mode | None |

## Interactive Commands

When in conversation mode, you can use these commands:

- `/help` - Show help message
- `/settings` - View/change generation settings
- `/clear` - Clear the screen
- `/reset` - Clear conversation context and start fresh
- `/status` - Show conversation status (exchanges, tokens)
- `/mode` - Toggle between conversation and single-prompt mode
- `/info` - Show model information
- `/load` - Load a different model
- `/quit` - Exit the program

## Examples

### Conversation Mode

```cmd
# Start conversation mode
run_cli.bat

# Example conversation:
💬 You: Hello! What's the weather like?
🤖 Assistant: I don't have access to real-time weather data, but I'd be happy to help you with weather-related questions or discuss weather patterns in general. What would you like to know?

� You: Tell me about thunderstorms
🤖 Assistant: Thunderstorms are fascinating weather phenomena! They form when warm, moist air rises rapidly through cooler air layers...

💬 You: How dangerous are they?
🤖 Assistant: [Continues the conversation with context from previous exchanges]
```

### Advanced Usage
```powershell
# Custom settings with PowerShell
.\run_cli.ps1 -Model "my_model.pt" -MaxTokens 200 -Temperature 0.9

# Single prompt with specific model
.\run_cli.ps1 -Model "pico_gpt_large_final.pt" -Prompt "In a distant galaxy" -MaxTokens 150
```

### Python Direct Usage
```python
python cli_client.py --model pico_gpt_large_best.pt --prompt "Hello world" --max-tokens 50 --temperature 0.7 --top-k 15
```

## Requirements

- Python 3.7 or higher
- PyTorch
- A trained Pico GPT model (`.pt` file)
- Windows 10/11 (for batch/PowerShell launchers)

## Model Files

The CLI client looks for these model files in order:
1. `pico_gpt_large_best.pt` (best model from large training)
2. `pico_gpt_large_final.pt` (final model from large training)
3. `pico_gpt_model.pt` (model from small training)
4. Any other `.pt` files in the directory

## Troubleshooting

### "Model file not found"
- Make sure you've trained a model first using `train_large.py` or `train_small.py`
- Check that the model file has a `.pt` extension
- Use the `--model` option to specify a custom path

### "CUDA out of memory"
- Use `--device cpu` to force CPU usage
- Reduce `--max-tokens` to generate shorter text
- Close other GPU-intensive applications

### "Python not found"
- Install Python from [python.org](https://python.org)
- Make sure Python is added to your PATH environment variable
- Restart your command prompt after installation

### PowerShell Execution Policy
If you get an execution policy error with PowerShell:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Performance Tips

- **🚀 GPU Usage**: The CLI automatically uses CUDA if available for faster generation
- **📏 Model Size**: Larger models generate better text but are slower
- **🌡️ Temperature**: Lower values (0.1-0.5) for more focused text, higher (0.8-1.2) for more creative text
- **🔝 Top-k**: Lower values (5-20) for more focused sampling, higher (40-100) for more diversity
- **💬 Conversation Length**: The CLI automatically manages context length to stay within model limits
- **🧠 Context Management**: Use `/reset` to clear conversation history if responses become inconsistent

## Integration

The CLI client can be integrated into other Windows workflows:

### Batch Scripts
```batch
@echo off
python cli_client.py --prompt "%1" --max-tokens 100 > output.txt
echo Generated text saved to output.txt
```

### PowerShell Functions
```powershell
function Generate-Text {
    param([string]$Prompt)
    python cli_client.py --prompt $Prompt --max-tokens 100
}
```

Enjoy using your Pico GPT model! 🤖

#!/usr/bin/env python3
"""
Fast Pico GPT CLI Client with optimized tokenizer
Uses the new fast tokenizer and improved model
"""

import torch
import argparse
import os
import sys
from pathlib import Path
try:
    import readline
except ImportError:
    readline = None
import atexit
from pico_gpt import GPT, GPTConfig
from fast_tokenizer import GPT2LikeTokenizer

# Fix Windows console encoding issues
if sys.platform == "win32":
    try:
        import locale
        locale.setlocale(locale.LC_ALL, '')
    except:
        pass


class FastPicoGPTCLI:
    def __init__(self, model_path='pico_gpt_fast.pt', device=None):
        """Initialize the fast CLI client"""
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.config = None
        self.history_file = Path.home() / '.pico_gpt_history'
        
        # Conversation state
        self.conversation_context = []
        self.max_conversation_length = 1024  # Tokens to keep in conversation
        self.conversation_active = False
        
        # Load readline history
        if readline:
            self.setup_readline()
    
    def setup_readline(self):
        """Setup readline for better command-line experience"""
        if not readline:
            return
        try:
            if self.history_file.exists():
                readline.read_history_file(str(self.history_file))
            readline.set_history_length(1000)
            atexit.register(readline.write_history_file, str(self.history_file))
        except Exception as e:
            print(f"Warning: Could not setup readline history: {e}")
    
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            print(f"[ERROR] Model file not found: {self.model_path}")
            print("Available model files:")
            model_files = [f for f in os.listdir('.') if f.endswith('.pt')]
            if model_files:
                for i, f in enumerate(model_files, 1):
                    size_mb = os.path.getsize(f) / (1024 * 1024)
                    print(f"  {i}. {f} ({size_mb:.1f} MB)")
                
                choice = input(f"\nSelect a model (1-{len(model_files)}) or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    sys.exit(0)
                
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(model_files):
                        self.model_path = model_files[idx]
                    else:
                        print("Invalid selection.")
                        sys.exit(1)
                except ValueError:
                    print("Invalid input.")
                    sys.exit(1)
            else:
                print("No model files found. Please train a model first.")
                sys.exit(1)
        
        print(f"[INFO] Loading model from {self.model_path}...")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Extract components from checkpoint
            self.config = checkpoint['config']
            self.tokenizer = checkpoint['tokenizer']
            
            # Create and load model
            self.model = GPT(self.config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"[SUCCESS] Fast model loaded!")
            print(f"   Device: {self.device}")
            print(f"   Parameters: {self.model.get_num_params():,}")
            print(f"   Vocab size: {self.config.vocab_size:,}")
            print(f"   Block size: {self.config.block_size}")
            
            # Show training info if available
            if 'best_val_loss' in checkpoint:
                print(f"   Best validation loss: {checkpoint['best_val_loss']:.4f}")
            if 'iter' in checkpoint:
                print(f"   Training iterations: {checkpoint['iter']:,}")
            
        except Exception as e:
            print(f"[ERROR] Error loading model: {e}")
            sys.exit(1)
    
    def generate_text(self, prompt, max_tokens=100, temperature=0.8, top_k=20, use_conversation=True):
        """Generate text from a prompt"""
        if not self.model or not self.tokenizer:
            print("[ERROR] Model not loaded!")
            return None, None
        
        try:
            # Prepare input text
            if use_conversation and self.conversation_active:
                # Use conversation context + new prompt
                full_text = "".join(self.conversation_context) + prompt
            else:
                full_text = prompt
            
            # Encode input
            input_tokens = self.tokenizer.encode(full_text)
            
            # Trim if too long for model's context window
            max_input_length = self.config.block_size - max_tokens - 1
            if len(input_tokens) > max_input_length:
                input_tokens = input_tokens[-max_input_length:]
                trimmed_text = self.tokenizer.decode(input_tokens)
                if use_conversation:
                    print(f"[INFO] Context trimmed to fit model's window ({max_input_length} tokens)")
            else:
                trimmed_text = full_text
            
            context = torch.tensor(input_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            
            # Generate
            with torch.no_grad():
                generated = self.model.generate(
                    context, 
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k
                )
            
            # Decode full generated text
            full_generated_text = self.tokenizer.decode(generated[0].tolist())
            
            # Extract only the new generated part
            new_text = full_generated_text[len(trimmed_text):]
            
            return new_text, full_generated_text
            
        except Exception as e:
            print(f"[ERROR] Error generating text: {e}")
            return None, None
    
    def update_conversation_context(self, user_input, model_response):
        """Update the conversation context with new exchange"""
        exchange = f"Human: {user_input}\nAssistant: {model_response}\n\n"
        self.conversation_context.append(exchange)
        
        # Keep conversation within reasonable token limits
        full_context = "".join(self.conversation_context)
        context_tokens = self.tokenizer.encode(full_context)
        
        # Trim old conversation if too long
        while len(context_tokens) > self.max_conversation_length and len(self.conversation_context) > 1:
            self.conversation_context.pop(0)
            full_context = "".join(self.conversation_context)
            context_tokens = self.tokenizer.encode(full_context)
    
    def clear_conversation(self):
        """Clear the conversation context"""
        self.conversation_context = []
        self.conversation_active = False
        print("[INFO] Conversation context cleared!")
    
    def show_conversation_status(self):
        """Show current conversation status"""
        if self.conversation_active:
            context_length = len(self.tokenizer.encode("".join(self.conversation_context)))
            exchanges = len(self.conversation_context)
            print(f"[STATUS] Conversation active: {exchanges} exchanges, {context_length} tokens")
        else:
            print("[STATUS] No active conversation (single-prompt mode)")
    
    def toggle_conversation_mode(self):
        """Toggle between conversation and single-prompt mode"""
        if self.conversation_active:
            self.conversation_active = False
            self.conversation_context = []
            print("[INFO] Switched to single-prompt mode")
        else:
            self.conversation_active = True
            print("[INFO] Switched to conversation mode")
            print("    Your inputs will now build context for ongoing conversation!")
    
    def print_banner(self):
        """Print welcome banner"""
        print("=" * 60)
        print("*** Fast Pico GPT CLI Client - Conversation Mode ***")
        print("=" * 60)
        print("Commands:")
        print("  /help         - Show this help")
        print("  /settings     - Show/change generation settings")
        print("  /clear        - Clear screen")
        print("  /reset        - Clear conversation context")
        print("  /status       - Show conversation status")
        print("  /mode         - Toggle conversation/single-prompt mode")
        print("  /quit         - Exit the program")
        print("  /load         - Load a different model")
        print("  /info         - Show model information")
        print("")
        print("[STATUS] Conversation Mode: Your messages build context for ongoing chat!")
        print("   Type your message and press Enter. The AI will remember context.")
        print("=" * 60)
    
    def show_settings(self):
        """Show current generation settings"""
        print("\n[SETTINGS] Current Generation Settings:")
        print(f"  Max tokens: {self.max_tokens}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Top-k: {self.top_k}")
        print(f"  Max conversation length: {self.max_conversation_length} tokens")
        print(f"  Conversation mode: {'ON' if self.conversation_active else 'OFF'}")
        print()
        
        # Ask if user wants to change settings
        change = input("Change settings? (y/N): ").strip().lower()
        if change == 'y':
            try:
                new_max = input(f"Max tokens ({self.max_tokens}): ").strip()
                if new_max:
                    self.max_tokens = int(new_max)
                
                new_temp = input(f"Temperature ({self.temperature}): ").strip()
                if new_temp:
                    self.temperature = float(new_temp)
                
                new_topk = input(f"Top-k ({self.top_k}): ").strip()
                if new_topk:
                    self.top_k = int(new_topk)
                
                new_conv_len = input(f"Max conversation length ({self.max_conversation_length}): ").strip()
                if new_conv_len:
                    self.max_conversation_length = int(new_conv_len)
                
                print("[SUCCESS] Settings updated!")
            except ValueError as e:
                print(f"[ERROR] Invalid input: {e}")
    
    def show_info(self):
        """Show model information"""
        if not self.model:
            print("[ERROR] No model loaded!")
            return
        
        print("\n[INFO] Model Information:")
        print(f"  Model file: {self.model_path}")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {self.model.get_num_params():,}")
        print(f"  Layers: {self.config.n_layer}")
        print(f"  Heads: {self.config.n_head}")
        print(f"  Embedding dim: {self.config.n_embd}")
        print(f"  Vocab size: {self.config.vocab_size:,}")
        print(f"  Block size: {self.config.block_size}")
        print(f"  Dropout: {self.config.dropout}")
        print(f"  Tokenizer: GPT-2 style (fast)")
        
        if torch.cuda.is_available() and self.device == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def run(self):
        """Main CLI loop"""
        # Load model
        self.load_model()
        
        # Default generation settings
        self.max_tokens = 100
        self.temperature = 0.8
        self.top_k = 20
        
        # Start in conversation mode by default
        self.conversation_active = True
        
        # Show banner
        self.print_banner()
        
        # Main loop
        try:
            while True:
                try:
                    # Show conversation indicator
                    if self.conversation_active:
                        prompt_prefix = "[You]: "
                    else:
                        prompt_prefix = "[Prompt]: "
                    
                    # Get user input
                    user_input = input(f"\n{prompt_prefix}").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle commands
                    if user_input.startswith('/'):
                        command = user_input[1:].lower()
                        
                        if command == 'help':
                            self.print_banner()
                        elif command == 'settings':
                            self.show_settings()
                        elif command == 'clear':
                            os.system('cls' if os.name == 'nt' else 'clear')
                        elif command == 'reset':
                            self.clear_conversation()
                        elif command == 'status':
                            self.show_conversation_status()
                        elif command == 'mode':
                            self.toggle_conversation_mode()
                        elif command == 'quit' or command == 'exit':
                            print("[INFO] Goodbye!")
                            break
                        elif command == 'load':
                            model_path = input("Enter model path: ").strip()
                            if model_path:
                                self.model_path = model_path
                                self.load_model()
                                self.clear_conversation()
                        elif command == 'info':
                            self.show_info()
                        else:
                            print(f"[ERROR] Unknown command: {command}")
                            print("Type /help for available commands.")
                        
                        continue
                    
                    # Generate text
                    if self.conversation_active:
                        print(f"[Assistant]: ", end="", flush=True)
                    else:
                        print(f"\n[INFO] Generating text (max_tokens={self.max_tokens}, temp={self.temperature}, top_k={self.top_k})...")
                    
                    new_text, full_text = self.generate_text(
                        user_input, 
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        use_conversation=self.conversation_active
                    )
                    
                    if new_text:
                        if self.conversation_active:
                            # In conversation mode, just show the response
                            print(new_text.strip())
                            
                            # Update conversation context
                            self.update_conversation_context(user_input, new_text.strip())
                        else:
                            # In single-prompt mode, show the full generated text
                            print(f"\n[OUTPUT] Generated text:")
                            print("-" * 40)
                            print(full_text)
                            print("-" * 40)
                    
                except KeyboardInterrupt:
                    print("\n\n[INFO] Goodbye!")
                    break
                except EOFError:
                    print("\n\n[INFO] Goodbye!")
                    break
                    
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Fast Pico GPT CLI Client')
    parser.add_argument('--model', '-m', default='pico_gpt_fast.pt',
                        help='Path to model file (default: pico_gpt_fast.pt)')
    parser.add_argument('--device', '-d', choices=['cpu', 'cuda', 'auto'], default='auto',
                        help='Device to use (default: auto)')
    parser.add_argument('--max-tokens', '-t', type=int, default=100,
                        help='Maximum tokens to generate (default: 100)')
    parser.add_argument('--temperature', '-T', type=float, default=0.8,
                        help='Temperature for generation (default: 0.8)')
    parser.add_argument('--top-k', '-k', type=int, default=20,
                        help='Top-k sampling (default: 20)')
    parser.add_argument('--prompt', '-p', type=str,
                        help='Single prompt mode (non-interactive)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Create CLI client
    cli = FastPicoGPTCLI(model_path=args.model, device=device)
    
    # Single prompt mode
    if args.prompt:
        cli.load_model()
        cli.max_tokens = args.max_tokens
        cli.temperature = args.temperature
        cli.top_k = args.top_k
        
        print(f"[INPUT] Prompt: {args.prompt}")
        new_text, full_text = cli.generate_text(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            use_conversation=False
        )
        
        if full_text:
            print(f"\n[OUTPUT] Generated:")
            print(full_text)
        return
    
    # Interactive mode
    cli.max_tokens = args.max_tokens
    cli.temperature = args.temperature
    cli.top_k = args.top_k
    cli.run()


if __name__ == '__main__':
    main()
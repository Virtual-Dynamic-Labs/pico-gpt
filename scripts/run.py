#!/usr/bin/env python3
"""
Simple runner for refactored Pico GPT
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_chat():
    """Run the chat interface"""
    from cli.cli_fast import FastPicoGPTCLI
    
    cli = FastPicoGPTCLI(model_path='models/pico_gpt_final.pt')
    
    # Quick test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', help='Test prompt')
    args, _ = parser.parse_known_args()
    
    if args.prompt:
        cli.load_model()
        new_text, full_text = cli.generate_text(args.prompt, max_tokens=50, use_conversation=False)
        print(f"Input: {args.prompt}")
        print(f"Output: {full_text}")
    else:
        cli.run()

if __name__ == '__main__':
    run_chat()
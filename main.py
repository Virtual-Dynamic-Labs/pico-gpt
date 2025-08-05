#!/usr/bin/env python3
"""
Pico GPT - Main entry point
"""

import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Pico GPT - A minimal GPT implementation')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Start interactive chat')
    chat_parser.add_argument('--model', '-m', default='models/pico_gpt_final.pt',
                           help='Path to model file')
    chat_parser.add_argument('--device', '-d', choices=['cpu', 'cuda', 'auto'], default='auto',
                           help='Device to use')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--type', choices=['fast', 'conversation', 'large'], 
                            default='conversation', help='Training type')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate text')
    gen_parser.add_argument('--prompt', '-p', required=True, help='Input prompt')
    gen_parser.add_argument('--model', '-m', default='models/pico_gpt_final.pt',
                          help='Path to model file')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--type', choices=['basic', 'conversation'], 
                           default='basic', help='Test type')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'chat':
        from cli.cli_fast import main as chat_main
        sys.argv = ['cli_fast.py']
        if args.model != 'models/pico_gpt_final.pt':
            sys.argv.extend(['--model', args.model])
        if args.device != 'auto':
            sys.argv.extend(['--device', args.device])
        os.chdir('cli')
        chat_main()
        
    elif args.command == 'train':
        if args.type == 'conversation':
            from training.train_final import train_final_conversation_model
            os.chdir('training')
            train_final_conversation_model()
        else:
            print(f"Training type '{args.type}' not implemented yet")
            
    elif args.command == 'generate':
        from cli.cli_fast import main as cli_main
        sys.argv = ['cli_fast.py', '--prompt', args.prompt]
        if args.model != 'models/pico_gpt_final.pt':
            sys.argv.extend(['--model', args.model])
        os.chdir('cli')
        cli_main()
        
    elif args.command == 'test':
        if args.type == 'basic':
            from tests.example import main as test_main
            os.chdir('tests')
            test_main()
        else:
            print(f"Test type '{args.type}' not implemented yet")

if __name__ == '__main__':
    main()
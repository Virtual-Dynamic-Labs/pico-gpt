#!/usr/bin/env python3
"""
Test script to verify conversation mode functionality
"""

import sys
import os
sys.path.append('.')
from cli_client import PicoGPTCLI

def test_conversation_mode():
    print("Testing Pico GPT Conversation Mode")
    print("=" * 50)
    
    # Create CLI instance
    cli = PicoGPTCLI(model_path='pico_gpt_large_best.pt', device='cuda')
    
    # Load model
    cli.load_model()
    
    # Set generation parameters
    cli.max_tokens = 50
    cli.temperature = 0.8
    cli.top_k = 20
    cli.conversation_active = True
    
    print("\n" + "=" * 50)
    print("Testing Conversation Context Building")
    print("=" * 50)
    
    # Test conversation flow
    test_exchanges = [
        "Hello, my name is Alice.",
        "What's my name?",
        "Tell me a short story about a cat.",
        "What was the story about?"
    ]
    
    for i, user_input in enumerate(test_exchanges, 1):
        print(f"\n--- Exchange {i} ---")
        print(f"User: {user_input}")
        
        # Generate response
        new_text, full_text = cli.generate_text(
            user_input,
            max_tokens=cli.max_tokens,
            temperature=cli.temperature,
            top_k=cli.top_k,
            use_conversation=True
        )
        
        if new_text:
            print(f"Assistant: {new_text.strip()}")
            
            # Update conversation context
            cli.update_conversation_context(user_input, new_text.strip())
            
            # Show context status
            context_length = len(cli.tokenizer.encode("".join(cli.conversation_context)))
            print(f"[Context: {len(cli.conversation_context)} exchanges, {context_length} tokens]")
        else:
            print("Assistant: [Error generating response]")
    
    print("\n" + "=" * 50)
    print("Testing Context Management")
    print("=" * 50)
    
    # Show full conversation context
    if cli.conversation_context:
        print("\nFull conversation history:")
        for i, exchange in enumerate(cli.conversation_context, 1):
            print(f"Exchange {i}: {repr(exchange[:100])}...")
    
    # Test context clearing
    print(f"\nBefore clear: {len(cli.conversation_context)} exchanges")
    cli.clear_conversation()
    print(f"After clear: {len(cli.conversation_context)} exchanges")
    
    print("\n" + "=" * 50)
    print("Testing Single-Prompt Mode")
    print("=" * 50)
    
    # Test single prompt mode
    cli.conversation_active = False
    single_prompt = "Once upon a time there was a dragon."
    
    print(f"Single prompt: {single_prompt}")
    new_text, full_text = cli.generate_text(
        single_prompt,
        max_tokens=cli.max_tokens,
        temperature=cli.temperature,
        top_k=cli.top_k,
        use_conversation=False
    )
    
    if full_text:
        print(f"Generated: {full_text}")
    
    print("\n" + "=" * 50)
    print("Conversation Mode Test Complete!")
    print("=" * 50)

if __name__ == "__main__":
    test_conversation_mode()
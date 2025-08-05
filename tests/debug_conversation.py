#!/usr/bin/env python3
"""
Debug script for conversation mode
"""

import torch
import sys
import os
from pico_gpt import GPT, GPTConfig
from tokenizer import SimpleTokenizer

def debug_generation():
    print("Debug: Loading model...")
    
    # Load model
    checkpoint = torch.load('pico_gpt_model.pt', map_location='cuda', weights_only=False)
    config = checkpoint['config']
    tokenizer = checkpoint['tokenizer']
    
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to('cuda')
    model.eval()
    
    print(f"Model loaded: {model.get_num_params():,} parameters")
    print(f"Block size: {config.block_size}, Vocab size: {config.vocab_size}")
    
    # Test simple generation
    test_prompt = "Hello, my name is Alice."
    print(f"\nTesting prompt: '{test_prompt}'")
    
    try:
        # Encode
        input_tokens = tokenizer.encode(test_prompt)
        print(f"Input tokens: {len(input_tokens)} tokens")
        print(f"First 10 tokens: {input_tokens[:10]}")
        
        # Create tensor
        context = torch.tensor(input_tokens, dtype=torch.long, device='cuda').unsqueeze(0)
        print(f"Context tensor shape: {context.shape}")
        
        # Generate
        with torch.no_grad():
            print("Generating...")
            original_length = context.shape[1]
            generated = model.generate(
                context, 
                max_new_tokens=20,
                temperature=0.8,
                top_k=20
            )
            print(f"Original length: {original_length}, Generated length: {generated.shape[1]}")
            print(f"New tokens generated: {generated.shape[1] - original_length}")
            
            # Check if we actually got new tokens
            if generated.shape[1] > original_length:
                # Decode full text
                full_text = tokenizer.decode(generated[0].tolist())
                new_tokens = generated[0][original_length:].tolist()
                new_text = tokenizer.decode(new_tokens)
                
                print(f"Full text: '{full_text}'")
                print(f"New text only: '{new_text}'")
                print(f"New token IDs: {new_tokens[:10]}")
                
                # Test decoding individual tokens
                print("Testing individual token decoding:")
                for i, token_id in enumerate(new_tokens[:5]):
                    try:
                        decoded = tokenizer.decode([token_id])
                        print(f"  Token {token_id}: '{decoded}' (repr: {repr(decoded)})")
                    except Exception as e:
                        print(f"  Token {token_id}: Error - {e}")
                
                # Check if tokens are valid
                print(f"Vocab size: {tokenizer.vocab_size}")
                print(f"Max token ID in new tokens: {max(new_tokens)}")
                print(f"Min token ID in new tokens: {min(new_tokens)}")
                
                # Check what these tokens actually are in the decoder
                print("Checking what tokens are in decoder:")
                for token_id in new_tokens[:5]:
                    if token_id in tokenizer.decoder:
                        token_str = tokenizer.decoder[token_id]
                        print(f"  {token_id}: '{token_str}'")
                    else:
                        print(f"  {token_id}: NOT IN DECODER")
            else:
                print("No new tokens were generated!")
            
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_generation()
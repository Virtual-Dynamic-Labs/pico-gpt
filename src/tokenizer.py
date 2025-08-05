import regex as re
import json
from typing import Dict, List


class SimpleTokenizer:
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.encoder = {}
        self.decoder = {}
        self._create_vocab()
    
    def _create_vocab(self):
        # Create a simple character-level tokenizer for demonstration
        # In practice, you'd use BPE or similar
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'-\n")
        
        # Add special tokens
        special_tokens = ["<|endoftext|>", "<|unk|>"]
        vocab = special_tokens + chars
        
        # Pad vocabulary to desired size
        while len(vocab) < self.vocab_size:
            vocab.append(f"<|pad_{len(vocab)}|>")
        
        self.encoder = {token: i for i, token in enumerate(vocab)}
        self.decoder = {i: token for i, token in enumerate(vocab)}
    
    def encode(self, text: str) -> List[int]:
        # Simple character-level encoding
        tokens = []
        for char in text:
            if char in self.encoder:
                tokens.append(self.encoder[char])
            else:
                tokens.append(self.encoder["<|unk|>"])  # Unknown token
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        # Simple character-level decoding
        text = ""
        for token in tokens:
            if token in self.decoder:
                char = self.decoder[token]
                if not char.startswith("<|") or char == "<|endoftext|>":
                    text += char
        return text


class BPETokenizer:
    """A minimal BPE tokenizer implementation"""
    
    def __init__(self, vocab_file: str = None):
        if vocab_file:
            self.load_vocab(vocab_file)
        else:
            self._create_default_vocab()
    
    def _create_default_vocab(self):
        # Create a basic vocabulary with byte-level BPE
        self.encoder = {}
        self.decoder = {}
        
        # Add all possible bytes
        vocab = list(range(256))
        
        # Add some common merges (simplified)
        common_pairs = [
            (ord(' '), ord('t')),  # ' t' -> common in English
            (ord('h'), ord('e')),  # 'he'
            (ord('i'), ord('n')),  # 'in'
            (ord('t'), ord('h')),  # 'th'
        ]
        
        for i, (a, b) in enumerate(common_pairs):
            vocab.append(256 + i)
        
        # Create encoder/decoder
        for i, token in enumerate(vocab):
            self.encoder[token] = i
            self.decoder[i] = token
    
    def encode(self, text: str) -> List[int]:
        # Convert to bytes and encode
        byte_tokens = text.encode('utf-8')
        return [self.encoder.get(b, 0) for b in byte_tokens]
    
    def decode(self, tokens: List[int]) -> str:
        # Decode tokens back to bytes, then to string
        try:
            byte_tokens = [self.decoder.get(t, 0) for t in tokens]
            return bytes(byte_tokens).decode('utf-8', errors='ignore')
        except:
            return ""
    
    def save_vocab(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.encoder, f)
    
    def load_vocab(self, filepath: str):
        with open(filepath, 'r') as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
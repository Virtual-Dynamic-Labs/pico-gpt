import regex as re
import json
from typing import Dict, List
from collections import Counter
import pickle


class FastWordTokenizer:
    """Fast word-level tokenizer optimized for English conversation"""
    
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.encoder = {}
        self.decoder = {}
        self.word_pattern = re.compile(r'\b\w+\b|[^\w\s]|\s+')
        
    def build_vocab_from_text(self, text: str):
        """Build vocabulary from training text - much faster than character-level"""
        print("Building word-level vocabulary...")
        
        # Tokenize into words and punctuation
        tokens = self.word_pattern.findall(text.lower())
        
        # Count frequencies
        word_counts = Counter(tokens)
        
        # Create base vocabulary
        vocab = ['<|pad|>', '<|unk|>', '<|bos|>', '<|eos|>']
        
        # Add most common words
        most_common = word_counts.most_common(self.vocab_size - len(vocab))
        vocab.extend([word for word, count in most_common])
        
        # Truncate to vocab_size
        vocab = vocab[:self.vocab_size]
        
        # Create encoder/decoder
        self.encoder = {word: i for i, word in enumerate(vocab)}
        self.decoder = {i: word for i, word in enumerate(vocab)}
        
        print(f"Vocabulary built: {len(vocab)} tokens")
        print(f"Coverage: {len(most_common)}/{len(word_counts)} unique words")
        
        # Show some examples
        common_words = list(vocab[4:20])  # Skip special tokens
        print(f"Common words: {common_words}")
        
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        tokens = self.word_pattern.findall(text.lower())
        return [self.encoder.get(token, self.encoder['<|unk|>']) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.decoder:
                token = self.decoder[token_id]
                if token not in ['<|pad|>', '<|bos|>', '<|eos|>']:
                    tokens.append(token)
        
        # Join tokens back to text
        result = ''.join(tokens)
        
        # Clean up spacing
        result = re.sub(r'\s+', ' ', result)  # Multiple spaces -> single space
        result = re.sub(r'\s+([.,!?;:])', r'\1', result)  # Fix punctuation spacing
        
        return result.strip()
    
    def save(self, filepath: str):
        """Save tokenizer to file"""
        data = {
            'vocab_size': self.vocab_size,
            'encoder': self.encoder,
            'decoder': self.decoder
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load tokenizer from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.vocab_size = data['vocab_size']
        self.encoder = data['encoder']
        self.decoder = data['decoder']


class GPT2LikeTokenizer:
    """Simplified GPT-2 style tokenizer for better performance"""
    
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.encoder = {}
        self.decoder = {}
        
        # GPT-2 style pattern for better tokenization
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
    def build_vocab_from_text(self, text: str):
        """Build vocabulary using GPT-2 style tokenization"""
        print("Building GPT-2 style vocabulary...")
        
        # Extract all tokens
        all_tokens = []
        for match in self.pattern.finditer(text):
            all_tokens.append(match.group())
        
        # Count frequencies
        token_counts = Counter(all_tokens)
        
        # Special tokens
        vocab = ['<|endoftext|>', '<|unk|>', '<|pad|>']
        
        # Add most frequent tokens
        most_common = token_counts.most_common(self.vocab_size - len(vocab))
        vocab.extend([token for token, count in most_common])
        
        # Truncate to vocab_size
        vocab = vocab[:self.vocab_size]
        
        # Create mappings
        self.encoder = {token: i for i, token in enumerate(vocab)}
        self.decoder = {i: token for i, token in enumerate(vocab)}
        
        print(f"GPT-2 style vocabulary: {len(vocab)} tokens")
        print(f"Sample tokens: {vocab[3:15]}")
        
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        tokens = []
        for match in self.pattern.finditer(text):
            token = match.group()
            tokens.append(self.encoder.get(token, self.encoder['<|unk|>']))
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.decoder:
                token = self.decoder[token_id]
                if not token.startswith('<|'):
                    tokens.append(token)
        return ''.join(tokens)
    
    def save(self, filepath: str):
        """Save tokenizer"""
        data = {
            'vocab_size': self.vocab_size,
            'encoder': self.encoder,
            'decoder': self.decoder
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load tokenizer"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.vocab_size = data['vocab_size']
        self.encoder = data['encoder']
        self.decoder = data['decoder']


def create_optimized_tokenizer(text_file: str, tokenizer_type: str = "gpt2", vocab_size: int = 8000):
    """Create and save an optimized tokenizer"""
    
    # Load text data
    print(f"Loading text data from {text_file}...")
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Text size: {len(text):,} characters")
    
    # Create tokenizer
    if tokenizer_type == "word":
        tokenizer = FastWordTokenizer(vocab_size=vocab_size)
    else:
        tokenizer = GPT2LikeTokenizer(vocab_size=vocab_size)
    
    # Build vocabulary
    tokenizer.build_vocab_from_text(text)
    
    # Test tokenization speed
    import time
    test_text = text[:10000]  # Test on first 10k chars
    
    start = time.time()
    tokens = tokenizer.encode(test_text)
    encoding_time = time.time() - start
    
    start = time.time()
    decoded = tokenizer.decode(tokens)
    decoding_time = time.time() - start
    
    print(f"\nPerformance Test:")
    print(f"Original length: {len(test_text)} chars")
    print(f"Tokenized length: {len(tokens)} tokens")
    print(f"Compression ratio: {len(test_text)/len(tokens):.2f}x")
    print(f"Encoding speed: {encoding_time*1000:.2f}ms")
    print(f"Decoding speed: {decoding_time*1000:.2f}ms")
    
    # Save tokenizer
    save_path = f"fast_tokenizer_{tokenizer_type}_{vocab_size}.pkl"
    tokenizer.save(save_path)
    print(f"Tokenizer saved to: {save_path}")
    
    return tokenizer


if __name__ == "__main__":
    # Create optimized tokenizers
    text_file = "data/combined_literature.txt"
    
    # Create GPT-2 style tokenizer (recommended)
    create_optimized_tokenizer(text_file, "gpt2", vocab_size=8000)
    
    # Create word-level tokenizer (alternative)
    create_optimized_tokenizer(text_file, "word", vocab_size=8000)
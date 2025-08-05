import os
import requests
import zipfile
import gzip
import json
from pathlib import Path
import time


def download_file(url, filename, description=""):
    """Download a file with progress tracking"""
    print(f"Downloading {description}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded_size = 0
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded_size += len(chunk)
                if total_size > 0:
                    progress = (downloaded_size / total_size) * 100
                    print(f"\r  Progress: {progress:.1f}% ({downloaded_size // 1024 // 1024} MB)", end='', flush=True)
    
    print(f"\n  Downloaded: {filename} ({downloaded_size // 1024 // 1024} MB)")


def download_openwebtext():
    """Download OpenWebText dataset (subset of data used to train GPT-2)"""
    print("=== Downloading OpenWebText Dataset ===")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # OpenWebText is large, so we'll download a manageable subset
    # Using the "openwebtext" dataset from HuggingFace
    
    urls = [
        "https://the-eye.eu/public/AI/pile_preliminary_components/openwebtext2.jsonl.zst.tar",
    ]
    
    # For now, let's use a smaller but high-quality dataset
    # We'll download the TinyStories dataset which is perfect for GPT training
    tiny_stories_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    
    try:
        filename = data_dir / "tinystories.tar.gz"
        if not filename.exists():
            download_file(tiny_stories_url, filename, "TinyStories Dataset")
            
            # Extract the archive
            print("Extracting dataset...")
            import tarfile
            with tarfile.open(filename, 'r:gz') as tar:
                tar.extractall(data_dir)
            print("  Extraction complete!")
        else:
            print(f"Dataset already exists: {filename}")
            
    except Exception as e:
        print(f"Error downloading TinyStories: {e}")
        print("Falling back to alternative datasets...")
        return download_alternative_datasets()
    
    return True


def download_alternative_datasets():
    """Download alternative high-quality text datasets"""
    print("=== Downloading Alternative Datasets ===")
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    datasets = []
    
    # 1. Project Gutenberg books
    gutenberg_books = [
        ("https://www.gutenberg.org/files/11/11-0.txt", "alice_in_wonderland.txt", "Alice in Wonderland"),
        ("https://www.gutenberg.org/files/1342/1342-0.txt", "pride_and_prejudice.txt", "Pride and Prejudice"),
        ("https://www.gutenberg.org/files/84/84-0.txt", "frankenstein.txt", "Frankenstein"),
        ("https://www.gutenberg.org/files/1661/1661-0.txt", "sherlock_holmes.txt", "Adventures of Sherlock Holmes"),
        ("https://www.gutenberg.org/files/2701/2701-0.txt", "moby_dick.txt", "Moby Dick"),
        ("https://www.gutenberg.org/files/345/345-0.txt", "dracula.txt", "Dracula"),
        ("https://www.gutenberg.org/files/76/76-0.txt", "huckleberry_finn.txt", "Huckleberry Finn"),
        ("https://www.gutenberg.org/files/1080/1080-0.txt", "modest_proposal.txt", "A Modest Proposal"),
        ("https://www.gutenberg.org/files/174/174-0.txt", "dorian_gray.txt", "The Picture of Dorian Gray"),
        ("https://www.gutenberg.org/files/1184/1184-0.txt", "count_monte_cristo.txt", "The Count of Monte Cristo"),
    ]
    
    print("Downloading classic literature...")
    for url, filename, title in gutenberg_books:
        filepath = data_dir / filename
        if not filepath.exists():
            try:
                download_file(url, filepath, title)
                datasets.append(filepath)
            except Exception as e:
                print(f"  Failed to download {title}: {e}")
        else:
            print(f"  Already exists: {title}")
            datasets.append(filepath)
    
    # 2. Create a combined dataset
    combined_file = data_dir / "combined_literature.txt"
    if not combined_file.exists() and datasets:
        print("\nCombining all texts into single training file...")
        with open(combined_file, 'w', encoding='utf-8') as outfile:
            for filepath in datasets:
                try:
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        # Clean up the text a bit
                        content = content.replace('\r\n', '\n')
                        content = content.replace('\r', '\n')
                        # Add separator between books
                        outfile.write(content)
                        outfile.write('\n\n' + '='*50 + '\n\n')
                except Exception as e:
                    print(f"  Error processing {filepath}: {e}")
        
        print(f"  Combined dataset created: {combined_file}")
        
        # Get file size
        size_mb = combined_file.stat().st_size / (1024 * 1024)
        print(f"  Total size: {size_mb:.1f} MB")
        
        return str(combined_file)
    
    return str(combined_file) if combined_file.exists() else None


def download_wikipedia_sample():
    """Download a sample of Wikipedia articles"""
    print("=== Downloading Wikipedia Sample ===")
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Use a pre-processed Wikipedia sample
    wiki_file = data_dir / "wikipedia_sample.txt"
    
    if not wiki_file.exists():
        # Create a substantial sample dataset by downloading multiple sources
        sources = [
            ("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", "shakespeare.txt"),
            ("https://raw.githubusercontent.com/karpathy/makemore/master/names.txt", "names.txt"),
        ]
        
        content_parts = []
        
        for url, name in sources:
            try:
                print(f"Downloading {name}...")
                response = requests.get(url)
                response.raise_for_status()
                content_parts.append(response.text)
                print(f"  Downloaded {name} ({len(response.text)} characters)")
            except Exception as e:
                print(f"  Failed to download {name}: {e}")
        
        if content_parts:
            combined_content = '\n\n'.join(content_parts)
            # Repeat content to make it larger
            combined_content = combined_content * 5
            
            with open(wiki_file, 'w', encoding='utf-8') as f:
                f.write(combined_content)
            
            size_mb = len(combined_content) / (1024 * 1024)
            print(f"  Created combined dataset: {size_mb:.1f} MB")
            
            return str(wiki_file)
    else:
        print(f"Wikipedia sample already exists: {wiki_file}")
        return str(wiki_file)
    
    return None


def main():
    print("Large Dataset Downloader for Pico GPT")
    print("=" * 50)
    
    # Try different dataset sources
    dataset_file = None
    
    # Option 1: Try TinyStories (best for GPT training)
    try:
        if download_openwebtext():
            # Check if we have extracted files
            data_dir = Path("data")
            txt_files = list(data_dir.glob("*.txt"))
            if txt_files:
                dataset_file = str(txt_files[0])  # Use first txt file found
    except Exception as e:
        print(f"TinyStories download failed: {e}")
    
    # Option 2: Fallback to classic literature
    if not dataset_file:
        dataset_file = download_alternative_datasets()
    
    # Option 3: Fallback to smaller samples
    if not dataset_file:
        dataset_file = download_wikipedia_sample()
    
    if dataset_file and os.path.exists(dataset_file):
        # Analyze the dataset
        with open(dataset_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        size_mb = len(content) / (1024 * 1024)
        num_chars = len(content)
        num_lines = content.count('\n')
        
        print(f"\n" + "=" * 50)
        print("DATASET READY!")
        print(f"=" * 50)
        print(f"File: {dataset_file}")
        print(f"Size: {size_mb:.1f} MB")
        print(f"Characters: {num_chars:,}")
        print(f"Lines: {num_lines:,}")
        print(f"Sample text (first 200 chars):")
        print("-" * 30)
        print(repr(content[:200]))
        print("-" * 30)
        
        # Create a symlink or copy for easy access
        main_dataset = Path("training_data.txt")
        if not main_dataset.exists():
            import shutil
            shutil.copy2(dataset_file, main_dataset)
            print(f"\nCopied to: {main_dataset}")
        
        print(f"\nYou can now train with: python train_large.py")
        
    else:
        print("\nError: Could not download any dataset!")
        print("You can manually place a large text file as 'training_data.txt'")


if __name__ == "__main__":
    main()
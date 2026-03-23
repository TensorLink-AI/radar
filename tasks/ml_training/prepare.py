"""
Data preparation script (frozen — do not modify).

Downloads the tiny Shakespeare dataset and tokenizes it into binary files
that train.py can memory-map for fast loading.

Outputs:
    data/train.bin   — uint16 numpy array of training token IDs
    data/val.bin     — uint16 numpy array of validation token IDs
    data/meta.json   — vocab size and tokenizer info

Usage:
    python prepare.py [--tokenizer {gpt2,char}]

By default uses GPT-2 BPE via tiktoken. Falls back to character-level
if tiktoken is not installed.
"""

import argparse
import json
import os
import struct
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
RAW_FILE = DATA_DIR / "input.txt"
TRAIN_SPLIT = 0.9


def download():
    """Download tiny Shakespeare if not already present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists():
        print(f"Raw data already exists at {RAW_FILE}")
        return RAW_FILE.read_text(encoding="utf-8")
    print(f"Downloading tiny Shakespeare from {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, RAW_FILE)
    text = RAW_FILE.read_text(encoding="utf-8")
    print(f"Downloaded {len(text):,} characters.")
    return text


def tokenize_gpt2(text: str):
    """Tokenize with GPT-2 BPE via tiktoken."""
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode_ordinary(text)
    vocab_size = enc.n_vocab  # 50257
    return tokens, vocab_size, "gpt2"


def tokenize_char(text: str):
    """Simple character-level tokenizer."""
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    tokens = [stoi[ch] for ch in text]
    vocab_size = len(chars)
    return tokens, vocab_size, "char"


def write_bin(path: Path, tokens: list[int]):
    """Write token IDs as a flat binary file of unsigned 16-bit integers."""
    with open(path, "wb") as f:
        for t in tokens:
            f.write(struct.pack("<H", t))
    print(f"Wrote {len(tokens):,} tokens to {path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare tiny Shakespeare dataset")
    parser.add_argument(
        "--tokenizer",
        choices=["gpt2", "char"],
        default="gpt2",
        help="Tokenizer to use (default: gpt2)",
    )
    args = parser.parse_args()

    text = download()

    # Tokenize
    tokenizer = args.tokenizer
    if tokenizer == "gpt2":
        try:
            tokens, vocab_size, tok_name = tokenize_gpt2(text)
        except ImportError:
            print("tiktoken not installed, falling back to character-level tokenizer.")
            tokens, vocab_size, tok_name = tokenize_char(text)
    else:
        tokens, vocab_size, tok_name = tokenize_char(text)

    print(f"Tokenizer: {tok_name} | Vocab size: {vocab_size:,} | Total tokens: {len(tokens):,}")

    # Split into train / val
    split_idx = int(len(tokens) * TRAIN_SPLIT)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    print(f"Train: {len(train_tokens):,} tokens | Val: {len(val_tokens):,} tokens")

    # Write binary files
    write_bin(DATA_DIR / "train.bin", train_tokens)
    write_bin(DATA_DIR / "val.bin", val_tokens)

    # Write metadata
    meta = {
        "vocab_size": vocab_size,
        "tokenizer": tok_name,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "total_chars": len(text),
    }
    meta_path = DATA_DIR / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata written to {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()

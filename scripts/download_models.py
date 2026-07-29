#!/usr/bin/env python3
"""
Download models from Hugging Face Hub.
Run once on first setup. Models saved to ./models/ (gitignored).
"""
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODELS = {
    "qwen2.5-3b-instruct-q4_k_m.gguf": {
        "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "size_gb": 2.2,
    },
    "bge-small-en-v1.5-f16.gguf": {
        "repo_id": "CompendiumLabs/bge-small-en-v1.5-gguf",
        "filename": "bge-small-en-v1.5-f16.gguf",
        "size_gb": 0.13,
    },
}


def download_model(name: str, info: dict) -> Path:
    dest = MODELS_DIR / name
    if dest.exists():
        print(f"[OK] {name} already exists at {dest}")
        return dest

    print(f"[DOWNLOAD] {name} ({info['size_gb']} GB) from {info['repo_id']}...")
    try:
        path = hf_hub_download(
            repo_id=info["repo_id"],
            filename=info["filename"],
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False,
        )
        print(f"[OK] Saved to {path}")
        return Path(path)
    except Exception as e:
        print(f"[ERROR] Failed to download {name}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    print(f"Models directory: {MODELS_DIR}")
    print("=" * 50)

    for name, info in MODELS.items():
        download_model(name, info)

    print("=" * 50)
    print("All models downloaded successfully!")
    print(f"Location: {MODELS_DIR}")


if __name__ == "__main__":
    main()
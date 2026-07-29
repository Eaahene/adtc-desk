#!/usr/bin/env python3
"""
Download llama-cli.exe pre-built binary from llama.cpp releases.
Run once on first setup. Binary saved to ./models/ (gitignored).
"""
import sys
from pathlib import Path
import urllib.request
import zipfile

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

LLAMA_CLI_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b4501/llama-b4501-win64-avx2.zip"
LLAMA_CLI_PATH = MODELS_DIR / "llama-cli.exe"


def download_file(url: str, dest: Path, desc: str = "") -> Path:
    """Download with simple progress."""
    if dest.exists():
        print(f"[OK] {dest.name} already exists at {dest}")
        return dest

    print(f"[DOWNLOAD] {desc or dest.name} from {url}...")

    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, block_num * block_size * 100 // total_size)
            bar_len = 40
            filled = bar_len * percent // 100
            bar = "#" * filled + "-" * (bar_len - filled)
            sys.stdout.write(f"\r  [{bar}] {percent}%")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
        print()
        print(f"[OK] Saved to {dest}")
        return dest
    except Exception as e:
        print(f"\n[ERROR] Failed: {e}", file=sys.stderr)
        if dest.exists():
            dest.unlink()
        sys.exit(1)


def extract_llama_cli(zip_path: Path):
    """Extract llama-cli.exe from the downloaded zip."""
    print(f"[EXTRACT] {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            if name.endswith("llama-cli.exe") or name.endswith("llama.exe"):
                zf.extract(name, MODELS_DIR)
                extracted = MODELS_DIR / name
                if extracted != LLAMA_CLI_PATH:
                    if LLAMA_CLI_PATH.exists():
                        LLAMA_CLI_PATH.unlink()
                    extracted.rename(LLAMA_CLI_PATH)
                print(f"[OK] Extracted llama-cli.exe to {LLAMA_CLI_PATH}")
                return
    print("[ERROR] Could not find llama-cli.exe in zip", file=sys.stderr)
    sys.exit(1)


def main():
    print(f"Models directory: {MODELS_DIR}")
    print("=" * 50)

    if LLAMA_CLI_PATH.exists():
        print(f"[OK] llama-cli.exe already exists at {LLAMA_CLI_PATH}")
        import subprocess
        result = subprocess.run([str(LLAMA_CLI_PATH), "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  Version: {result.stdout.strip()}")
        print("=" * 50)
        print("Ready!")
        return

    zip_path = MODELS_DIR / "llama.zip"
    download_file(LLAMA_CLI_URL, zip_path, "llama.cpp binaries")
    extract_llama_cli(zip_path)
    zip_path.unlink()

    import subprocess
    result = subprocess.run([str(LLAMA_CLI_PATH), "--version"], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[OK] Verified: {result.stdout.strip()}")
    else:
        print(f"[ERROR] Verification failed: {result.stderr}")
        sys.exit(1)

    print("=" * 50)
    print("llama-cli.exe ready!")


if __name__ == "__main__":
    main()
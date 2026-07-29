#!/bin/bash
# download_model.sh — Downloads Qwen2.5-1.5B-Instruct GGUF model for ADTC 2026
# Idempotent: skips download if file already exists

set -e

MODEL_DIR="model"
MODEL_FILE="qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"

echo "=== Otimi Desk — Model Download ==="

# Create model directory if it doesn't exist
mkdir -p "${MODEL_DIR}"

# Check if model already exists
if [ -f "${MODEL_PATH}" ]; then
    echo "Model already exists at ${MODEL_PATH}"
    echo "Skipping download."
    exit 0
fi

echo "Downloading ${MODEL_FILE}..."
echo "URL: ${MODEL_URL}"
echo "Destination: ${MODEL_PATH}"

# Download with curl (follow redirects, show progress)
if command -v curl &> /dev/null; then
    curl -L -o "${MODEL_PATH}" "${MODEL_URL}"
elif command -v wget &> /dev/null; then
    wget -O "${MODEL_PATH}" "${MODEL_URL}"
else
    echo "Error: Neither curl nor wget found. Please install one of them."
    exit 1
fi

# Verify download
if [ -f "${MODEL_PATH}" ]; then
    FILE_SIZE=$(stat -f%z "${MODEL_PATH}" 2>/dev/null || stat -c%s "${MODEL_PATH}" 2>/dev/null || echo "unknown")
    echo ""
    echo "Download complete!"
    echo "File: ${MODEL_PATH}"
    echo "Size: ${FILE_SIZE} bytes"
else
    echo "Error: Download failed. File not found at ${MODEL_PATH}"
    exit 1
fi
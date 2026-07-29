"""
Compare TPS between 1.5B and 3B models.
"""
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_cpp import Llama

MODELS_DIR = Path(__file__).parent.parent / "models"

def test_model(model_path, n_threads=6, n_ctx=2048):
    """Test a specific model and return TPS."""
    print(f"\nTesting: {model_path.name}")
    print(f"  Threads: {n_threads}, Context: {n_ctx}")
    
    start_load = time.time()
    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=64,
        verbose=False,
        use_mmap=True,
        use_mlock=True,
        n_gpu_layers=0,
    )
    load_time = time.time() - start_load
    print(f"  Load time: {load_time:.2f}s")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Reply in one sentence."},
        {"role": "user", "content": "What is the capital of Ghana?"}
    ]
    
    # Warm up
    llm.create_chat_completion(messages=messages, max_tokens=10)
    
    # Measure TPS
    start_gen = time.time()
    result = llm.create_chat_completion(
        messages=messages,
        max_tokens=128,
        temperature=0.1,
        top_p=0.95,
    )
    end_gen = time.time()
    
    elapsed = end_gen - start_gen
    response = result["choices"][0]["message"]["content"]
    token_count = len(response.split())
    tps = token_count / elapsed if elapsed > 0 else 0
    
    print(f"  Generated: {token_count} tokens in {elapsed:.2f}s")
    print(f"  TPS: {tps:.2f}")
    print(f"  Response: {response[:80]}...")
    
    return tps

if __name__ == "__main__":
    # Test 3B model
    tps_3b = test_model(MODELS_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf")
    
    # Test 1.5B model
    tps_1_5b = test_model(MODELS_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf")
    
    print("\n" + "="*50)
    print("COMPARISON")
    print("="*50)
    print(f"3B model TPS:  {tps_3b:.2f}")
    print(f"1.5B model TPS: {tps_1_5b:.2f}")
    print(f"Speedup: {tps_1_5b/tps_3b:.2f}x")

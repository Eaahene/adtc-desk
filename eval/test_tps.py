"""
Simple TPS test for measuring throughput with mlock optimization.
"""
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.llm import LocalLLM

def measure_tps():
    """Measure tokens per second for generation."""
    print("Loading model with mlock enabled...")
    llm = LocalLLM(verbose=False)
    
    # Simple prompt for TPS measurement
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Reply in one sentence."},
        {"role": "user", "content": "What is the capital of Ghana?"}
    ]
    
    # Warm up
    print("Warming up...")
    llm.chat(messages, max_tokens=10)
    
    # Measure TPS
    print("Measuring TPS...")
    start_time = time.time()
    result = llm.chat(messages, max_tokens=128)
    end_time = time.time()
    
    elapsed = end_time - start_time
    # Rough token count (split by spaces as approximation)
    token_count = len(result.split())
    
    tps = token_count / elapsed if elapsed > 0 else 0
    
    print(f"\nResults:")
    print(f"  Generated: {token_count} tokens")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  TPS: {tps:.2f}")
    print(f"  Response: {result[:100]}...")
    
    return tps

if __name__ == "__main__":
    tps = measure_tps()

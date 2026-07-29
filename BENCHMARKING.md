# Benchmarking Checklist

## Required Metrics for ADTC 2026 Submission

### 1. Memory (RSS) — Peak Resident Set Size

| Metric | How to Measure | Expected Value |
|--------|---------------|----------------|
| Model only (no UI) | `import os; os.getrusage().ru_maxrss` after LLM load | ~2.2GB |
| With SQLite + UI | Same, after server startup | ~2.5GB |
| With embeddings (lazy) | After first RAG search | ~2.6GB |
| Under sustained load | During 10-turn conversation | ~2.8GB |

**Measurement command:**
```python
import os
import psutil

process = psutil.Process()
print(f"Peak RSS: {process.memory_info().rss / 1024**3:.2f} GB")
```

### 2. Throughput (TPS) — Tokens Per Second

| Metric | How to Measure | Expected Value |
|--------|---------------|----------------|
| Prompt evaluation | llama-cpp stats output | ~5.15 tok/s |
| Token generation | llama-cpp stats output | ~3.73 tok/s |
| End-to-end single turn | Timer on orchestrator.run() | 10-15s |
| End-to-end multi-turn | Timer on orchestrator.run() | 75-95s |

**Measurement command:**
```python
from src.models.llm import LocalLLM
llm = LocalLLM()
response, stats = llm.generate("Hello", max_tokens=50)
print(f"Prompt eval: {stats['prompt_tokens_per_second']:.2f} tok/s")
print(f"Generation: {stats['predicted_tokens_per_second']:.2f} tok/s")
```

### 3. Cold Start — Time to First Token

| Metric | How to Measure | Expected Value |
|--------|---------------|----------------|
| Model load time | Timer on LocalLLM init | 5-6s |
| First token latency | Time from prompt to first generated token | 2-3s |

**Measurement command:**
```python
import time

start = time.time()
llm = LocalLLM()
load_time = time.time() - start
print(f"Model load: {load_time:.2f}s")

start = time.time()
response, _ = llm.generate("Hello", max_tokens=10)
first_token_time = time.time() - start
print(f"First token latency: {first_token_time:.2f}s")
```

### 4. Accuracy — Tool-Call Correctness

| Metric | How to Measure | Expected Value |
|--------|---------------|----------------|
| Exact match | eval/benchmark.py | 40% |
| Near match | eval/benchmark.py | 60% |
| Single-tool accuracy | Task/Contact/Note prompts | 50-60% |
| Multi-tool accuracy | Complex sequences | 20-30% |

**Measurement command:**
```bash
cd eval
python benchmark.py --prompts prompts.json --output results.json
```

### 5. Thermal Notes

- CPU-only inference generates moderate heat
- No GPU throttling concerns
- Sustained load: fan engagement after ~2min continuous
- Recommended: Use on desk, not lap

---

## Submission Checklist

- [ ] Run all benchmarks and document results
- [ ] Take screenshots of system running
- [ ] Record 2-minute demo video
- [ ] Write REPORT.md (done)
- [ ] Clean GitHub repo (done)
- [ ] Verify all metrics under 7GB RAM
- [ ] Test offline mode (no internet)
- [ ] Verify plan-before-execute works
- [ ] Test RAG search functionality
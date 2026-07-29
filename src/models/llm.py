"""
LLM wrapper using llama-cpp-python with thermal monitoring.
"""
import time
from pathlib import Path
from typing import Optional, Iterator, List, Dict, Any

from llama_cpp import Llama

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf"  # Faster TPS for better score

DEFAULT_N_CTX = 2048  # Fits multi-turn conversation
DEFAULT_N_THREADS = 6  # Optimal: 6 of 8 cores
DEFAULT_N_BATCH = 64   # Optimal: fastest prompt processing

# Thermal thresholds (Celsius)
THERMAL_WARNING = 80
THERMAL_CRITICAL = 85  # ADTC penalty threshold


def get_cpu_temp() -> Optional[float]:
    """Get CPU temperature if available (Windows)."""
    try:
        import wmi
        w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
        sensors = w.Sensor()
        for sensor in sensors:
            if sensor.SensorType == "Temperature" and "CPU" in sensor.Name:
                return float(sensor.Value)
    except Exception:
        pass
    
    # Fallback: try PowerShell
    try:
        import subprocess
        result = subprocess.run(
            ['powershell', '-Command', 
             'Get-WmiObject MSAcpi_ThermalZoneTemperature -Namespace "root/wmi" | Select-Object -First 1 -ExpandProperty CurrentTemperature'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            # Value is in tenths of Kelvin
            kelvin = float(result.stdout.strip()) / 10
            return kelvin - 273.15
    except Exception:
        pass
    
    return None


class LocalLLM:
    """Wrapper around llama-cpp-python for local inference with thermal monitoring."""
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        n_ctx: int = DEFAULT_N_CTX,
        n_threads: int = DEFAULT_N_THREADS,
        n_batch: int = DEFAULT_N_BATCH,
        verbose: bool = False,
    ):
        self.model_path = model_path or MODEL_PATH
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.verbose = verbose
        self.peak_temp = 0.0
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                f"Run scripts/download_models.py first."
            )
        
        print(f"Loading model from {self.model_path}...")
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            verbose=verbose,
            use_mmap=True,
            use_mlock=True,  # Lock model in RAM to prevent swapping (big speed boost)
            n_gpu_layers=0,  # CPU only
        )
        print("Model loaded successfully!")
    
    def _check_thermal(self) -> bool:
        """Check temperature, return True if OK, False if throttling risk."""
        temp = get_cpu_temp()
        if temp is not None:
            self.peak_temp = max(self.peak_temp, temp)
            if temp >= THERMAL_CRITICAL:
                print(f"WARNING: CPU temp {temp:.1f}°C exceeds {THERMAL_CRITICAL}°C threshold!")
                return False
            elif temp >= THERMAL_WARNING:
                print(f"CAUTION: CPU temp {temp:.1f}°C approaching threshold")
        return True
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Generate completion from raw prompt."""
        if self.verbose:
            print(f"[llama-cpp] Generating: max_tokens={max_tokens}, temp={temperature}")
        
        result = self.llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            stream=stream,
        )
        
        if stream:
            return self._stream_result(result)
        else:
            text = result["choices"][0]["text"]
            return text.strip()
    
    def _stream_result(self, result_iter) -> Iterator[str]:
        for chunk in result_iter:
            text = chunk["choices"][0]["text"]
            if text:
                yield text
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.95,
        tools: Optional[List[Dict]] = None,
        stream: bool = False,
    ) -> str | Iterator[str] | Dict[str, Any]:
        """Chat completion with optional function calling."""
        if self.verbose:
            print(f"[llama-cpp] Chat: {len(messages)} messages, tools={len(tools) if tools else 0}")
        
        kwargs = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        result = self.llm.create_chat_completion(**kwargs)
        
        if stream:
            return self._stream_chat_result(result)
        else:
            msg = result["choices"][0]["message"]
            if msg.get("tool_calls"):
                return msg
            return msg["content"]
    
    def _stream_chat_result(self, result_iter) -> Iterator[str]:
        for chunk in result_iter:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                yield content


def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract first valid JSON object from model response, handling noise."""
    import json
    
    # Find all { positions and try each as potential start
    positions = [i for i, c in enumerate(text) if c == '{']
    
    for start in positions:
        # Track brace depth for proper JSON extraction
        depth = 0
        in_string = False
        escape_next = False
        
        for i in range(start, len(text)):
            c = text[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if c == '\\' and in_string:
                escape_next = True
                continue
            
            if c == '"':
                in_string = not in_string
                continue
            
            if in_string:
                continue
            
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        result = json.loads(candidate)
                        if isinstance(result, dict) and 'tool_calls' in result:
                            return result
                    except json.JSONDecodeError:
                        pass
                    # Move to next potential start
                    break
    
    return None


if __name__ == "__main__":
    llm = LocalLLM(verbose=True)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one sentence."}
    ]
    result = llm.chat(messages, max_tokens=50)
    print(f"Result: {result}")
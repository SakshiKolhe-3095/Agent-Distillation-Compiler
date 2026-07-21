# serve.py API Contract -- Week 7 Thu

## Class: HybridModelServer (`inference/serve.py`)

### Constructor
```python
HybridModelServer(
    router_path: str = "models/router.pkl",
    student_path: str = "models/sakshi_llama31_8b_rank4/checkpoint-9",
    teacher_path: str = "models/qlora-primary-dpo/checkpoint-7",
    max_seq_length: int = 512,
)
```
Loads the router immediately. Loads the student model immediately. Teacher
model is loaded lazily on first use (saves VRAM).

### Method: generate()
```python
server.generate(problem: str, max_new_tokens: int = 512) -> dict
```

**Input:**
- `problem` (str): Natural language description of the coding problem.
- `max_new_tokens` (int, optional): Max tokens to generate. Default 512.

**Output (dict):**
```json
{
  "code": "def add(a, b):\n    return a + b",
  "route": "student"
}
```
- `code`: Generated Python code string (no markdown fences).
- `route`: Which model answered -- `"student"` or `"teacher"`.

### Routing logic
The `ComplexityRouter` (`inference/router.py`) classifies each problem
based on length, loop count, function count, and line count. Simple
problems route to the student (fast, local Llama3.1-8B). Complex problems
route to the DPO-aligned teacher (Qwen2.5-7B, higher quality).

### Usage example (for backend/main.py)
```python
from inference.serve import HybridModelServer

server = HybridModelServer()  # load once at startup

# per-request call
result = server.generate("Write a function that finds the nth Fibonacci number.")
print(result["code"])   # the generated code
print(result["route"])  # "student" or "teacher"
```

### Error handling
- If `router_path` or `student_path` not found: raises `FileNotFoundError` at init.
- If `teacher_path` not found: raises `FileNotFoundError` on first teacher-routed request (lazy load).
- Generation errors bubble up as exceptions -- caller should wrap in try/except.

### Performance notes
- Student model: ~31 t/s generation (RTX 4050, CUDA, 4-bit quantized)
- Teacher model: loaded lazily -- first teacher call adds ~30-60s load time
- Both models cannot fit in VRAM simultaneously on a 6GB card -- the lazy
  load approach means switching routes mid-session will cause a reload
  penalty. For production, pre-load both on a machine with more VRAM.
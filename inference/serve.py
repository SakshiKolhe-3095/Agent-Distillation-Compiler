# """
# Model serving skeleton.
# Loads a HF checkpoint (or later GGUF) and exposes generate().
# """

# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# class ModelServer:
#     def __init__(self, model_path: str, device: str = None):
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
#         ).to(self.device)
#         self.model.eval()

#     def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
#         with torch.no_grad():
#             output = self.model.generate(
#                 **inputs,
#                 max_new_tokens=max_new_tokens,
#                 temperature=temperature,
#                 do_sample=temperature > 0,
#                 pad_token_id=self.tokenizer.eos_token_id,
#             )
#         text = self.tokenizer.decode(output[0], skip_special_tokens=True)
#         return text[len(prompt):].strip()


# if __name__ == "__main__":
#     # quick manual test
#     server = ModelServer("gpt2")  # replace with real checkpoint later
#     print(server.generate("Write a function that adds two numbers."))

"""
Model serving skeleton.
Loads a HF checkpoint or a GGUF checkpoint (via llama-cpp-python, using the
CUDA/Vulkan builds from Week 1) and exposes a unified generate() interface.
"""

import torch


class ModelServer:
    """
    Unified interface over two backends:
      - HuggingFace transformers (for .safetensors / HF-format checkpoints)
      - llama.cpp via llama-cpp-python (for .gguf checkpoints)
    Picks the backend automatically based on the file extension / path.
    """

    def __init__(self, model_path: str, device: str = None, n_gpu_layers: int = -1):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if model_path.lower().endswith(".gguf"):
            self.backend = "gguf"
            self._init_gguf(model_path, n_gpu_layers)
        else:
            self.backend = "hf"
            self._init_hf(model_path)

    def _init_hf(self, model_path: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def _init_gguf(self, model_path: str, n_gpu_layers: int):
        from llama_cpp import Llama
        # n_gpu_layers=-1 offloads all layers to GPU if available (CUDA build);
        # set to 0 to force CPU-only.
        self.llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, verbose=False)

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        if self.backend == "gguf":
            return self._generate_gguf(prompt, max_new_tokens, temperature)
        return self._generate_hf(prompt, max_new_tokens, temperature)

    def _generate_hf(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return text[len(prompt):].strip()

    def _generate_gguf(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        result = self.llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            echo=False,
        )
        return result["choices"][0]["text"].strip()


if __name__ == "__main__":
    # quick manual test (HF backend, small model for a fast smoke test)
    server = ModelServer("gpt2")
    print(server.generate("Write a function that adds two numbers."))
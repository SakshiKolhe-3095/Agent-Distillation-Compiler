"""
Model serving skeleton.
Loads a HF checkpoint (or later GGUF) and exposes generate().
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ModelServer:
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
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


if __name__ == "__main__":
    # quick manual test
    server = ModelServer("gpt2")  # replace with real checkpoint later
    print(server.generate("Write a function that adds two numbers."))
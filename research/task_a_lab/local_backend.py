"""Optional local Transformers backend. Never downloads weights or calls an API.

Only mocks are exercised in this task; actual model inference is unverified.
"""

from pathlib import Path


class LocalTransformersBackend:
    def __init__(self, model_path, device="cpu", max_input_tokens=4096, max_new_tokens=1024):
        path = Path(model_path)
        if not path.is_dir():
            raise ValueError("Use an existing local model directory; hub IDs are not accepted")
        if device not in {"cpu", "cuda"} or min(max_input_tokens, max_new_tokens) < 1:
            raise ValueError("Invalid device or token budget")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA explicitly requested but unavailable")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.max_input_tokens, self.max_new_tokens = max_input_tokens, max_new_tokens
        context_limit = getattr(self.model.config, "max_position_embeddings", None)
        self.context_limit = context_limit if isinstance(context_limit, int) else None

    def generate(self, messages):
        text = self.tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False, truncation=False)
        length = encoded["input_ids"].shape[-1]
        if length > self.max_input_tokens or self.context_limit and length + self.max_new_tokens > self.context_limit:
            raise ValueError("Input exceeds token budget; context is never silently truncated")
        encoded = {k: value.to(self.device) for k, value in encoded.items()}
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        with self.torch.inference_mode():
            generated = self.model.generate(**encoded, do_sample=False, max_new_tokens=self.max_new_tokens,
                                            pad_token_id=pad_id)
        return self.tokenizer.decode(generated[0, length:], skip_special_tokens=True)

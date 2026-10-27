import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import transformers
import numpy as np
from typing import List, Tuple, Dict, Union

class SentimentCausalLM:
    """
    Wrapper for Causal LM sentiment classification on GPU/CPU.
    Calculates probabilities for positive/negative sentiment labels.
    """
    def __init__(
        self, 
        model_name: str = "microsoft/phi-2", 
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = None
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        if model_name.strip() in ["microsoft/phi", "phi", "phi-2", "phi2"]:
            model_name = "microsoft/phi-2"

        print(f"Loading model '{model_name}' on {self.device}...")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        config = transformers.AutoConfig.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
            config.pad_token_id = self.tokenizer.pad_token_id

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch_dtype if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        if self.device == "cuda" and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Determine target token IDs for " positive" and " negative"
        # We test both with and without leading space to get exact token IDs
        pos_id = self._get_token_id(" positive", default="positive")
        neg_id = self._get_token_id(" negative", default="negative")
        
        self.target_token_ids = [neg_id, pos_id] # 0: negative, 1: positive
        print(f"Target token IDs -> Negative: {neg_id} ('{self.tokenizer.decode([neg_id])}'), Positive: {pos_id} ('{self.tokenizer.decode([pos_id])}')")

        # Hardware-invariant complexity tracking
        self.forward_pass_count = 0
        self.backward_pass_count = 0
        self.forward_token_count = 0
        self._baseline_forward_latency_s = None

    def _get_token_id(self, token_str: str, default: str) -> int:
        ids = self.tokenizer.encode(token_str, add_special_tokens=False)
        if len(ids) > 0:
            return ids[0]
        ids = self.tokenizer.encode(default, add_special_tokens=False)
        return ids[0]

    def format_prompt(self, text: str) -> str:
        """Formats review text into a sentiment classification prompt."""
        return f"Review: {text}\nSentiment:"

    def predict_single(self, text: str) -> Tuple[int, np.ndarray]:
        """
        Predicts sentiment for a single text.
        Returns:
            predicted_class (0 for negative, 1 for positive),
            probabilities array [p_neg, p_pos]
        """
        prompt = self.format_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self.forward_pass_count += 1
        self.forward_token_count += int(inputs["input_ids"].shape[1])
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            target_logits = next_token_logits[self.target_token_ids]
            probs = torch.softmax(target_logits, dim=-1).float().cpu().numpy()
            
        pred_class = int(np.argmax(probs))
        return pred_class, probs

    def predict_tokens_batch(self, token_ids_batch: np.ndarray, sub_batch_size: int = None) -> np.ndarray:
        """
        Memory-safe batch prediction function compatible with SHAP and Occlusion.
        Takes 1D or 2D array of token IDs, returns probabilities [p_neg, p_pos].
        Dynamically adjusts sub_batch_size to fully utilize high-VRAM GPUs (A100/L4).
        """
        if sub_batch_size is None:
            if self.device == "cuda" and torch.cuda.is_available():
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                sub_batch_size = 128 if total_mem >= 30 else (64 if total_mem >= 16 else 16)
            else:
                sub_batch_size = 8
        if len(token_ids_batch.shape) == 1:
            token_ids_batch = np.expand_dims(token_ids_batch, axis=0)
            
        num_samples = len(token_ids_batch)
        self.forward_pass_count += num_samples
        self.forward_token_count += int(token_ids_batch.size)

        if num_samples <= sub_batch_size:
            token_ids_tensor = torch.tensor(token_ids_batch, dtype=torch.long, device=self.device)
            attention_mask = (token_ids_tensor != self.pad_token_id).long()
            with torch.no_grad():
                outputs = self.model(input_ids=token_ids_tensor, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]
                target_logits = next_token_logits[:, self.target_token_ids]
                probs = torch.softmax(target_logits, dim=-1).float().cpu().numpy()
            del token_ids_tensor, attention_mask, outputs, next_token_logits, target_logits
            return probs

        probs_list = []
        for i in range(0, num_samples, sub_batch_size):
            chunk = token_ids_batch[i:i + sub_batch_size]
            token_ids_tensor = torch.tensor(chunk, dtype=torch.long, device=self.device)
            attention_mask = (token_ids_tensor != self.pad_token_id).long()
            with torch.no_grad():
                outputs = self.model(input_ids=token_ids_tensor, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]
                target_logits = next_token_logits[:, self.target_token_ids]
                p = torch.softmax(target_logits, dim=-1).float().cpu().numpy()
                probs_list.append(p)
            del token_ids_tensor, attention_mask, outputs, next_token_logits, target_logits
            
        return np.vstack(probs_list)

    def predict_texts_batch(self, texts: List[str], sub_batch_size: int = 16) -> np.ndarray:
        """
        Batched prediction for raw texts with automatic left-padding and attention mask.
        Returns array of shape (len(texts), 2) with probabilities [p_neg, p_pos].
        """
        prompts = [self.format_prompt(t) for t in texts]
        num_samples = len(prompts)
        probs_list = []

        for i in range(0, num_samples, sub_batch_size):
            chunk = prompts[i:i + sub_batch_size]
            encoded = self.tokenizer(
                chunk, 
                padding=True, 
                return_tensors="pt"
            ).to(self.device)
            
            self.forward_pass_count += len(chunk)
            self.forward_token_count += int(encoded["input_ids"].numel())

            with torch.no_grad():
                outputs = self.model(**encoded)
                next_token_logits = outputs.logits[:, -1, :]
                target_logits = next_token_logits[:, self.target_token_ids]
                p = torch.softmax(target_logits, dim=-1).float().cpu().numpy()
                probs_list.append(p)
            del encoded, outputs, next_token_logits, target_logits

        return np.vstack(probs_list)

    def get_input_embeddings(self) -> torch.nn.Module:
        """Returns the input embedding layer of the underlying model."""
        return self.model.get_input_embeddings()

    def predict_embeddings(
        self, 
        inputs_embeds: torch.Tensor, 
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Calculates probabilities directly from input embeddings.
        Essential for the Embedding Masking Operator in faithfulness evaluations.
        """
        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2], 
                dtype=torch.long, 
                device=inputs_embeds.device
            )

        self.forward_pass_count += int(inputs_embeds.shape[0])
        self.forward_token_count += int(inputs_embeds.shape[0] * inputs_embeds.shape[1])

        with torch.no_grad():
            outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            next_token_logits = outputs.logits[:, -1, :]
            target_logits = next_token_logits[:, self.target_token_ids]
            probs = torch.softmax(target_logits, dim=-1).float()
        return probs

    def reset_counters(self):
        """Resets hardware-invariant operation counters."""
        self.forward_pass_count = 0
        self.backward_pass_count = 0
        self.forward_token_count = 0

    def increment_backward_passes(self, count: int = 1):
        """Increments backward / gradient pass counter."""
        self.backward_pass_count += count

    def get_counters(self) -> Dict[str, int]:
        """Returns snapshot of hardware-invariant counters."""
        return {
            "forward_passes": self.forward_pass_count,
            "backward_passes": self.backward_pass_count,
            "forward_tokens": self.forward_token_count
        }

    def benchmark_baseline_forward(self, num_trials: int = 15) -> float:
        """
        Benchmarks median latency of 1 forward pass on current hardware.
        Used to normalize all explanation runtimes into hardware-invariant Forward Units (FwdU).
        """
        import time
        if self._baseline_forward_latency_s is not None:
            return self._baseline_forward_latency_s

        sample_prompt = self.format_prompt("It is a charming and thoroughly engaging romantic comedy.")
        inputs = self.tokenizer(sample_prompt, return_tensors="pt").to(self.device)

        # Warm-up GPU
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(**inputs)

        latencies = []
        for _ in range(num_trials):
            if self.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = self.model(**inputs)
            if self.device == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        self._baseline_forward_latency_s = float(np.median(latencies))
        return self._baseline_forward_latency_s

    def compute_relative_fwdu(self, elapsed_seconds: float) -> float:
        """Converts raw elapsed seconds into Relative Forward Units (FwdU)."""
        baseline = self.benchmark_baseline_forward()
        return float(elapsed_seconds / baseline) if baseline > 0 else 0.0


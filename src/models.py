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

        # Determine target token IDs for " positive" and " negative"
        # We test both with and without leading space to get exact token IDs
        pos_id = self._get_token_id(" positive", default="positive")
        neg_id = self._get_token_id(" negative", default="negative")
        
        self.target_token_ids = [neg_id, pos_id] # 0: negative, 1: positive
        print(f"Target token IDs -> Negative: {neg_id} ('{self.tokenizer.decode([neg_id])}'), Positive: {pos_id} ('{self.tokenizer.decode([pos_id])}')")

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
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            target_logits = next_token_logits[self.target_token_ids]
            probs = torch.softmax(target_logits, dim=-1).cpu().numpy()
            
        pred_class = int(np.argmax(probs))
        return pred_class, probs

    def predict_tokens_batch(self, token_ids_batch: np.ndarray, sub_batch_size: int = 8) -> np.ndarray:
        """
        Memory-safe batch prediction function compatible with SHAP and Occlusion.
        Takes 1D or 2D array of token IDs, returns probabilities [p_neg, p_pos].
        Uses sub-batching to prevent CUDA Out-Of-Memory on 8GB GPUs.
        """
        if len(token_ids_batch.shape) == 1:
            token_ids_batch = np.expand_dims(token_ids_batch, axis=0)
            
        num_samples = len(token_ids_batch)
        if num_samples <= sub_batch_size:
            token_ids_tensor = torch.tensor(token_ids_batch, dtype=torch.long, device=self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=token_ids_tensor)
                next_token_logits = outputs.logits[:, -1, :]
                target_logits = next_token_logits[:, self.target_token_ids]
                probs = torch.softmax(target_logits, dim=-1).float().cpu().numpy()
            del token_ids_tensor, outputs, next_token_logits, target_logits
            return probs

        probs_list = []
        for i in range(0, num_samples, sub_batch_size):
            chunk = token_ids_batch[i:i + sub_batch_size]
            token_ids_tensor = torch.tensor(chunk, dtype=torch.long, device=self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=token_ids_tensor)
                next_token_logits = outputs.logits[:, -1, :]
                target_logits = next_token_logits[:, self.target_token_ids]
                p = torch.softmax(target_logits, dim=-1).float().cpu().numpy()
                probs_list.append(p)
            del token_ids_tensor, outputs, next_token_logits, target_logits
            
        return np.vstack(probs_list)

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import transformers
import numpy as np
from typing import List, Tuple, Dict, Union, Optional

DEFAULT_CALIBRATION_TEXTS: Tuple[str, ...] = ("N/A", "", "[MASK]")


class SentimentCausalLM:
    """
    Wrapper for Causal LM sentiment classification on GPU/CPU.
    Calculates probabilities for positive/negative sentiment labels with optional
    Contextual Calibration (Zhao et al., ICML 2021) to remove zero-shot label prior bias.
    """
    def __init__(
        self, 
        model_name: str = "microsoft/phi-2", 
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = None,
        sub_batch_size: Optional[int] = None,
        calibrate_logits: bool = True,
        calibration_texts: Optional[List[str]] = None,
        prompt_prefix: str = "Review: ",
        prompt_suffix: str = "\nSentiment:",
        neg_label: str = " negative",
        pos_label: str = " positive",
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        if model_name.strip() in ["microsoft/phi", "phi", "phi-2", "phi2"]:
            model_name = "microsoft/phi-2"
        self.model_name = model_name
        self.dtype = torch_dtype if self.device == "cuda" else torch.float32

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
        # Freeze model weights so autograd does not retain intermediate activations for dL/dW
        # during torch.autograd.grad(..., inputs_embeds) in IG and Gradient x Input (~1.5-2 GB VRAM saved on 7B-8B).
        self.model.requires_grad_(False)
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

        # Cache parameter count and default sub-batch size once
        self.param_count = sum(p.numel() for p in self.model.parameters())
        self.max_tokens_per_batch: Optional[int] = None
        if self.device == "cuda" and torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            is_heavy_model = self.param_count > 6e9
            if total_mem >= 30:
                self.max_tokens_per_batch = 8192
            elif total_mem >= 16:
                self.max_tokens_per_batch = 768 if is_heavy_model else 4096
            else:
                self.max_tokens_per_batch = 512 if is_heavy_model else 2048

            if sub_batch_size is not None:
                self.default_sub_batch_size = max(1, int(sub_batch_size))
            elif total_mem >= 30:
                self.default_sub_batch_size = 128
            elif total_mem >= 16:
                self.default_sub_batch_size = 24 if is_heavy_model else 64
            else:
                self.default_sub_batch_size = 16 if is_heavy_model else 32
        else:
            self.default_sub_batch_size = max(1, int(sub_batch_size)) if sub_batch_size is not None else 8

        # Target token IDs and prompt template configuration
        self.calibrate_logits: bool = bool(calibrate_logits)
        self._calibration_texts: Optional[List[str]] = list(calibration_texts) if calibration_texts is not None else None
        self._calibration_bias: Optional[torch.Tensor] = None
        self.calibration_prior_probs: Optional[np.ndarray] = None
        self.set_prompt_template(
            prompt_prefix=prompt_prefix,
            prompt_suffix=prompt_suffix,
            neg_label=neg_label,
            pos_label=pos_label,
            calibrate_logits=self.calibrate_logits,
            calibration_texts=self._calibration_texts,
        )

        # Hardware-invariant complexity tracking
        self.forward_pass_count = 0
        self.backward_pass_count = 0
        self.forward_token_count = 0
        self._baseline_forward_latency_s = None

    def _compute_calibration_bias(self, calibration_texts: Optional[List[str]] = None) -> None:
        """
        Estimates the model's content-free label prior over {'negative', 'positive'} using
        neutral placeholders ('N/A', '', '[MASK]') per Zhao et al. (ICML 2021) and stores
        the centered log-prior vector b_cal so calibrated logits equal z(x) - b_cal.
        """
        cf_inputs = list(calibration_texts) if calibration_texts is not None else list(DEFAULT_CALIBRATION_TEXTS)
        self._calibration_bias = None
        cf_probs = []
        with torch.inference_mode():
            for cf_text in cf_inputs:
                ids = self.encode_prompt(cf_text, return_tensors="pt")
                raw_logits = self.forward_target_logits(input_ids=ids)[0]
                p = torch.softmax(raw_logits.float(), dim=-1).cpu().numpy()
                cf_probs.append(p)
        p_bar = np.clip(np.mean(cf_probs, axis=0), 1e-8, 1.0 - 1e-8)
        p_bar = p_bar / np.sum(p_bar)
        log_p = np.log(p_bar)
        b_cal = log_p - float(np.mean(log_p))
        self.calibration_prior_probs = p_bar.astype(np.float64)
        self._calibration_bias = torch.tensor(b_cal, dtype=torch.float32, device=self.device).view(1, 2)
        margin_shift = float(b_cal[0] - b_cal[1])
        print(
            f"Contextual calibration (Zhao et al., 2021) -> content-free prior "
            f"P(neg)={p_bar[0]:.4f}, P(pos)={p_bar[1]:.4f} | pos-neg logit correction: {margin_shift:+.4f}"
        )

    @property
    def device_name(self) -> str:
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "CPU"

    @property
    def device_memory_gb(self) -> float:
        if self.device == "cuda" and torch.cuda.is_available():
            return round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        return 0.0

    def _get_token_id(self, token_str: str, default: str) -> int:
        ids = self.tokenizer.encode(token_str, add_special_tokens=False)
        if len(ids) > 0:
            return ids[0]
        ids = self.tokenizer.encode(default, add_special_tokens=False)
        return ids[0]

    def set_prompt_template(
        self,
        prompt_prefix: str = "Review: ",
        prompt_suffix: str = "\nSentiment:",
        neg_label: str = " negative",
        pos_label: str = " positive",
        calibrate_logits: Optional[bool] = None,
        calibration_texts: Optional[List[str]] = None,
    ) -> None:
        """
        Updates the prompt template and verbalizers in-place and recomputes
        Contextual Calibration bias without reloading model weights.
        """
        self._prefix_str = str(prompt_prefix)
        self._suffix_str = str(prompt_suffix)
        self._prefix_ids = self.tokenizer.encode(self._prefix_str, add_special_tokens=False)
        self._suffix_ids = self.tokenizer.encode(self._suffix_str, add_special_tokens=False)

        neg_id = self._get_token_id(neg_label, default=neg_label.strip())
        pos_id = self._get_token_id(pos_label, default=pos_label.strip())
        self.target_token_ids = [neg_id, pos_id]

        if calibrate_logits is not None:
            self.calibrate_logits = bool(calibrate_logits)
        if calibration_texts is not None:
            self._calibration_texts = list(calibration_texts)

        if self.calibrate_logits:
            self._compute_calibration_bias(self._calibration_texts)
        else:
            self._calibration_bias = None
            self.calibration_prior_probs = None

    def format_prompt(self, text: str) -> str:
        """Formats review text into a sentiment classification prompt."""
        return f"{self._prefix_str}{text}{self._suffix_str}"

    def encode_prompt(self, text: str, return_tensors: Optional[str] = "pt") -> Union[torch.Tensor, np.ndarray, List[int]]:
        """
        Canonical tokenizer entrypoint for a single review text.
        Always sets add_special_tokens=False so token indices match extract_word_units across all models.
        """
        prompt = self.format_prompt(text)
        encoded = self.tokenizer(prompt, return_tensors=return_tensors, add_special_tokens=False)
        input_ids = encoded["input_ids"]
        if return_tensors == "pt":
            return input_ids.to(self.device)
        return input_ids

    def find_review_span(self, text: str, num_tokens: Optional[int] = None) -> Tuple[int, int, List[int]]:
        """
        Finds the [start, end) token span of the review inside the formatted prompt
        along with the full token ID list (with add_special_tokens=False).
        """
        prompt = self.format_prompt(text)
        text_start = len(self._prefix_str)
        text_end = text_start + len(text)

        if getattr(self.tokenizer, "is_fast", False):
            try:
                encoded = self.tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
                input_ids = list(encoded["input_ids"])
                n_tok = len(input_ids)
                matching = [
                    i for i, (s, e) in enumerate(encoded["offset_mapping"])
                    if max(s, text_start) < min(e, text_end)
                ]
                if matching and (num_tokens is None or n_tok == num_tokens):
                    return matching[0], matching[-1] + 1, input_ids
            except Exception:
                pass

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        n_tok = len(input_ids) if num_tokens is None else num_tokens
        p_len = len(self._prefix_ids)
        s_len = len(self._suffix_ids)
        if n_tok > p_len + s_len:
            return p_len, n_tok - s_len, input_ids
        return 0, n_tok, input_ids

    def forward_target_logits(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes float32 logits of shape (batch_size, 2) for [negative, positive] at the final token position
        WITHOUT materializing full-vocabulary logits across all sequence positions (B, L, V).
        Projects the 2-class head in float32 to eliminate bfloat16 logit quantization (0.0625-0.125 ULP steps)
        and applies Contextual Calibration bias removal when enabled.
        """
        base = getattr(self.model, "model", None) or getattr(self.model, "transformer", None)
        out_emb = self.model.get_output_embeddings()
        if base is not None and out_emb is not None and hasattr(out_emb, "weight"):
            base_out = base(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            last_hidden = base_out.last_hidden_state[:, -1, :].to(torch.float32)  # (B, H)
            target_w = out_emb.weight[self.target_token_ids].to(torch.float32)    # (2, H)
            raw_b = getattr(out_emb, "bias", None)
            target_b = raw_b[self.target_token_ids].to(torch.float32) if raw_b is not None else None
            logits = torch.nn.functional.linear(last_hidden, target_w, target_b)
        else:
            outputs = self.model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits[:, -1, self.target_token_ids].to(torch.float32)

        cal_bias = getattr(self, "_calibration_bias", None)
        if cal_bias is not None:
            logits = logits - cal_bias.to(device=logits.device, dtype=logits.dtype)
        return logits

    def predict_single(self, text: str) -> Tuple[int, np.ndarray]:
        """
        Predicts sentiment for a single text.
        Returns:
            predicted_class (0 for negative, 1 for positive),
            probabilities array [p_neg, p_pos]
        """
        input_ids = self.encode_prompt(text, return_tensors="pt")
        self.forward_pass_count += 1
        self.forward_token_count += int(input_ids.shape[1])
        
        with torch.inference_mode():
            target_logits = self.forward_target_logits(input_ids=input_ids)[0]
            probs = torch.softmax(target_logits.float(), dim=-1).cpu().numpy()
            
        pred_class = int(np.argmax(probs))
        return pred_class, probs

    def _effective_sub_batch_size(self, requested_bs: Optional[int], seq_len: int) -> int:
        """
        Caps sub-batch size by a dynamic token-per-batch budget so long sequences
        do not spill VRAM into Windows Shared GPU Memory on 24 GB cards.
        """
        bs = self.default_sub_batch_size if requested_bs is None else max(1, int(requested_bs))
        max_tokens = getattr(self, "max_tokens_per_batch", None)
        if max_tokens is not None and seq_len > 0:
            cap = max(4, max_tokens // int(seq_len))
            bs = min(bs, cap)
        return max(1, bs)

    def predict_tokens_batch(self, token_ids_batch: np.ndarray, sub_batch_size: int = None) -> np.ndarray:
        """
        Memory-safe batch prediction function compatible with SHAP and Occlusion.
        Takes 1D or 2D array of token IDs, returns probabilities [p_neg, p_pos].
        """
        if len(token_ids_batch.shape) == 1:
            token_ids_batch = np.expand_dims(token_ids_batch, axis=0)

        seq_len = int(token_ids_batch.shape[1])
        sub_batch_size = self._effective_sub_batch_size(sub_batch_size, seq_len)

        num_samples = len(token_ids_batch)
        self.forward_pass_count += num_samples
        self.forward_token_count += int(token_ids_batch.size)

        if num_samples <= sub_batch_size:
            token_ids_tensor = torch.as_tensor(token_ids_batch, dtype=torch.long, device=self.device)
            attention_mask = (token_ids_tensor != self.pad_token_id).long()
            with torch.inference_mode():
                target_logits = self.forward_target_logits(input_ids=token_ids_tensor, attention_mask=attention_mask)
                probs = torch.softmax(target_logits.float(), dim=-1).cpu().numpy()
            del token_ids_tensor, attention_mask, target_logits
            return probs

        probs_list = []
        for i in range(0, num_samples, sub_batch_size):
            chunk = token_ids_batch[i:i + sub_batch_size]
            token_ids_tensor = torch.as_tensor(chunk, dtype=torch.long, device=self.device)
            attention_mask = (token_ids_tensor != self.pad_token_id).long()
            with torch.inference_mode():
                target_logits = self.forward_target_logits(input_ids=token_ids_tensor, attention_mask=attention_mask)
                p = torch.softmax(target_logits.float(), dim=-1).cpu().numpy()
                probs_list.append(p)
            del token_ids_tensor, attention_mask, target_logits
            
        return np.vstack(probs_list)

    def predict_texts_batch(self, texts: List[str], sub_batch_size: int = None) -> np.ndarray:
        """
        Batched prediction for raw texts with automatic left-padding and attention mask.
        Returns array of shape (len(texts), 2) with probabilities [p_neg, p_pos].
        """
        if not texts:
            return np.zeros((0, 2), dtype=np.float32)

        prompts = [self.format_prompt(t) for t in texts]
        approx_seq_len = max(16, max(len(p) for p in prompts) // 3)
        sub_batch_size = self._effective_sub_batch_size(sub_batch_size, approx_seq_len)
        num_samples = len(prompts)
        probs_list = []

        for i in range(0, num_samples, sub_batch_size):
            chunk = prompts[i:i + sub_batch_size]
            encoded = self.tokenizer(
                chunk, 
                padding=True, 
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.device)
            
            self.forward_pass_count += len(chunk)
            self.forward_token_count += int(encoded["input_ids"].numel())

            with torch.inference_mode():
                target_logits = self.forward_target_logits(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"]
                )
                p = torch.softmax(target_logits.float(), dim=-1).cpu().numpy()
                probs_list.append(p)
            del encoded, target_logits

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

        with torch.inference_mode():
            target_logits = self.forward_target_logits(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            probs = torch.softmax(target_logits.float(), dim=-1)
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


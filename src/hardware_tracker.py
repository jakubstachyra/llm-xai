import os
import time
import subprocess
import threading
import platform
import pandas as pd
from typing import Optional, Dict, Any, List

try:
    import torch
except ImportError:
    torch = None


class GpuTelemetrySampler(threading.Thread):
    """
    Lightweight background thread that samples power draw (W) and GPU utilization (%)
    periodically via nvidia-smi without stalling the main execution thread.
    """
    def __init__(self, interval_s: float = 2.0):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.power_draw_samples: List[float] = []
        self.gpu_util_samples: List[float] = []
        self.power_limit_w: float = 0.0

    def run(self):
        while not self.stop_event.is_set():
            try:
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw,power.limit,utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1.0
                )
                if res.returncode == 0:
                    parts = [p.strip() for p in res.stdout.strip().split(",")]
                    if len(parts) >= 3:
                        p_draw = float(parts[0]) if parts[0] != "[N/A]" else 0.0
                        p_lim = float(parts[1]) if parts[1] != "[N/A]" else 0.0
                        util = float(parts[2]) if parts[2] != "[N/A]" else 0.0

                        if p_draw > 0:
                            self.power_draw_samples.append(p_draw)
                        if p_lim > 0:
                            self.power_limit_w = p_lim
                        self.gpu_util_samples.append(util)
            except Exception:
                pass
            self.stop_event.wait(self.interval_s)

    def stop(self):
        self.stop_event.set()
        self.join(timeout=1.0)


class HardwareSessionTracker:
    """
    Hardware session profiler for LLM-XAI experiments.
    Tracks wall-clock runtime, network passes (forward/backward), token volume,
    peak VRAM utilization (allocated & reserved pool), power draw (W), and GPU utilization (%)
    across different compute environments (e.g., RTX 4060 Laptop vs RTX 4090 vs A100).
    """

    def __init__(self, model_wrapper, experiment_name: str, num_samples: Optional[int] = None):
        self.model_wrapper = model_wrapper
        self.experiment_name = experiment_name
        self.num_samples = num_samples
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.start_counters: Dict[str, int] = {}
        self.end_counters: Dict[str, int] = {}
        self.elapsed_s: float = 0.0
        self.sampler: Optional[GpuTelemetrySampler] = None

    @property
    def device_name(self) -> str:
        if hasattr(self.model_wrapper, "device_name"):
            return self.model_wrapper.device_name
        if torch is not None and torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "CPU"

    @property
    def device_memory_gb(self) -> float:
        if hasattr(self.model_wrapper, "device_memory_gb"):
            return self.model_wrapper.device_memory_gb
        if torch is not None and torch.cuda.is_available():
            return round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        return 0.0

    def start(self):
        """Starts timing, resets peak memory stats, and launches background GPU telemetry."""
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        self.start_time = time.perf_counter()
        if hasattr(self.model_wrapper, "get_counters"):
            self.start_counters = self.model_wrapper.get_counters()
        else:
            self.start_counters = {"forward_passes": 0, "backward_passes": 0, "forward_tokens": 0}

        # Start non-blocking GPU telemetry sampler
        if torch is not None and torch.cuda.is_available():
            self.sampler = GpuTelemetrySampler(interval_s=2.0)
            self.sampler.start()

    def stop(self, num_samples: Optional[int] = None) -> Dict[str, Any]:
        """Stops timing, stops sampler, and calculates metrics delta."""
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()

        self.end_time = time.perf_counter()
        if num_samples is not None:
            self.num_samples = num_samples

        self.elapsed_s = self.end_time - (self.start_time if self.start_time else self.end_time)

        # Stop telemetry sampler
        peak_power_w = 0.0
        avg_power_w = 0.0
        power_limit_w = 0.0
        peak_gpu_util = 0.0
        avg_gpu_util = 0.0

        if self.sampler is not None:
            self.sampler.stop()
            if self.sampler.power_draw_samples:
                peak_power_w = round(max(self.sampler.power_draw_samples), 1)
                avg_power_w = round(sum(self.sampler.power_draw_samples) / len(self.sampler.power_draw_samples), 1)
            power_limit_w = round(self.sampler.power_limit_w, 1)
            if self.sampler.gpu_util_samples:
                peak_gpu_util = round(max(self.sampler.gpu_util_samples), 1)
                avg_gpu_util = round(sum(self.sampler.gpu_util_samples) / len(self.sampler.gpu_util_samples), 1)

        if hasattr(self.model_wrapper, "get_counters"):
            self.end_counters = self.model_wrapper.get_counters()
        else:
            self.end_counters = {"forward_passes": 0, "backward_passes": 0, "forward_tokens": 0}

        fwd_delta = self.end_counters.get("forward_passes", 0) - self.start_counters.get("forward_passes", 0)
        bwd_delta = self.end_counters.get("backward_passes", 0) - self.start_counters.get("backward_passes", 0)
        tok_delta = self.end_counters.get("forward_tokens", 0) - self.start_counters.get("forward_tokens", 0)

        peak_vram_mb = 0.0
        vram_reserved_mb = 0.0
        if torch is not None and torch.cuda.is_available():
            peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
            vram_reserved_mb = round(torch.cuda.max_memory_reserved() / (1024 * 1024), 2)

        hours = int(self.elapsed_s // 3600)
        minutes = int((self.elapsed_s % 3600) // 60)
        seconds = self.elapsed_s % 60
        formatted_duration = f"{hours}h {minutes:02d}m {seconds:04.1f}s" if hours > 0 else f"{minutes}m {seconds:04.1f}s"

        fwd_per_s = round(fwd_delta / self.elapsed_s, 2) if self.elapsed_s > 0 else 0.0

        model_name = getattr(self.model_wrapper, "model_name", "Unknown")
        dtype_str = str(getattr(self.model_wrapper, "dtype", "Unknown")).replace("torch.", "")

        stats = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_name": self.experiment_name,
            "model_name": model_name,
            "dtype": dtype_str,
            "device_name": self.device_name,
            "vram_total_gb": self.device_memory_gb,
            "num_samples": self.num_samples if self.num_samples is not None else 0,
            "duration_s": round(self.elapsed_s, 2),
            "duration_formatted": formatted_duration,
            "forward_passes": fwd_delta,
            "backward_passes": bwd_delta,
            "forward_tokens": tok_delta,
            "peak_vram_mb": peak_vram_mb,
            "vram_reserved_mb": vram_reserved_mb,
            "peak_power_w": peak_power_w,
            "avg_power_w": avg_power_w,
            "power_limit_w": power_limit_w,
            "gpu_util_pct": peak_gpu_util,
            "avg_gpu_util_pct": avg_gpu_util,
            "fwd_per_second": fwd_per_s
        }
        return stats

    def snapshot_stats(self, num_samples: int) -> Dict[str, Any]:
        """Non-destructively computes live hardware telemetry mid-run (for checkpoint flushing)."""
        now = time.perf_counter()
        elapsed = now - (self.start_time if self.start_time else now)
        end_c = self.model_wrapper.get_counters() if hasattr(self.model_wrapper, "get_counters") else {}
        fwd_delta = end_c.get("forward_passes", 0) - self.start_counters.get("forward_passes", 0)
        bwd_delta = end_c.get("backward_passes", 0) - self.start_counters.get("backward_passes", 0)
        tok_delta = end_c.get("forward_tokens", 0) - self.start_counters.get("forward_tokens", 0)

        peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if (torch is not None and torch.cuda.is_available()) else 0.0
        vram_reserved_mb = round(torch.cuda.max_memory_reserved() / (1024 * 1024), 2) if (torch is not None and torch.cuda.is_available()) else 0.0

        peak_power_w = round(max(self.sampler.power_draw_samples), 1) if (self.sampler and self.sampler.power_draw_samples) else 0.0
        avg_power_w = round(sum(self.sampler.power_draw_samples) / len(self.sampler.power_draw_samples), 1) if (self.sampler and self.sampler.power_draw_samples) else 0.0
        power_limit_w = round(self.sampler.power_limit_w, 1) if self.sampler else 0.0
        peak_gpu_util = round(max(self.sampler.gpu_util_samples), 1) if (self.sampler and self.sampler.gpu_util_samples) else 0.0
        avg_gpu_util = round(sum(self.sampler.gpu_util_samples) / len(self.sampler.gpu_util_samples), 1) if (self.sampler and self.sampler.gpu_util_samples) else 0.0

        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = elapsed % 60
        formatted_duration = f"{hours}h {minutes:02d}m {seconds:04.1f}s" if hours > 0 else f"{minutes}m {seconds:04.1f}s"

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_name": self.experiment_name,
            "model_name": getattr(self.model_wrapper, "model_name", "Unknown"),
            "dtype": str(getattr(self.model_wrapper, "dtype", "Unknown")).replace("torch.", ""),
            "device_name": self.device_name,
            "vram_total_gb": self.device_memory_gb,
            "num_samples": num_samples,
            "duration_s": round(elapsed, 2),
            "duration_formatted": formatted_duration,
            "forward_passes": fwd_delta,
            "backward_passes": bwd_delta,
            "forward_tokens": tok_delta,
            "peak_vram_mb": peak_vram_mb,
            "vram_reserved_mb": vram_reserved_mb,
            "peak_power_w": peak_power_w,
            "avg_power_w": avg_power_w,
            "power_limit_w": power_limit_w,
            "gpu_util_pct": peak_gpu_util,
            "avg_gpu_util_pct": avg_gpu_util,
            "fwd_per_second": round(fwd_delta / elapsed, 2) if elapsed > 0 else 0.0
        }

    def log_session(
        self,
        output_csv: str = "results/hardware_session_log.csv",
        stats: Optional[Dict[str, Any]] = None,
        extra_csv_paths: Optional[List[str]] = None
    ):
        """Appends session statistics to the centralized hardware session CSV and any Google Drive paths."""
        if stats is None:
            stats = self.stop()

        paths = [output_csv]
        if extra_csv_paths:
            for p in extra_csv_paths:
                if p and os.path.abspath(p) != os.path.abspath(output_csv):
                    paths.append(p)

        df_new = pd.DataFrame([stats])
        for target_csv in paths:
            os.makedirs(os.path.dirname(os.path.abspath(target_csv)), exist_ok=True)
            if os.path.exists(target_csv):
                try:
                    df_existing = pd.read_csv(target_csv)
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    df_combined.to_csv(target_csv, index=False)
                except Exception:
                    df_new.to_csv(target_csv, mode="a", header=False, index=False)
            else:
                df_new.to_csv(target_csv, index=False)

    def print_summary(self, stats: Optional[Dict[str, Any]] = None):
        """Prints a human-readable ASCII summary box to the console."""
        if stats is None:
            stats = self.stop()

        dev_str = f"{stats['device_name']}"
        if stats['vram_total_gb'] > 0:
            dev_str += f" ({stats['vram_total_gb']} GB VRAM)"

        print("\n" + "=" * 74)
        print(" RAPORT PROFILOWANIA SPRZĘTOWEGO (HARDWARE & COMPUTATIONAL COST)")
        print("=" * 74)
        print(f" Eksperyment:          {stats['experiment_name']}")
        print(f" Urządzenie:           {dev_str}")
        print(f" Liczba próbek:        {stats['num_samples']}")
        print(f" Czas trwania sesji:   {stats['duration_formatted']} ({stats['duration_s']} s)")
        print(f" Przejścia w przód:    {stats['forward_passes']:,} forward passes ({stats['fwd_per_second']:.2f} pass/s)")
        print(f" Przejścia w tył:      {stats['backward_passes']:,} backward autograd passes")
        print(f" Wolumen tokenów:      {stats['forward_tokens']:,} tokenów w przetworzonych sekwencjach")
        print(f" VRAM Zaalokowany:     {stats['peak_vram_mb']:.1f} MB")
        if stats.get('vram_reserved_mb', 0) > 0:
            print(f" VRAM Zarezerwowany:   {stats['vram_reserved_mb']:.1f} MB (PyTorch Memory Pool)")
        if stats.get('peak_power_w', 0) > 0:
            p_lim = f" / {stats['power_limit_w']:.0f} W" if stats.get('power_limit_w', 0) > 0 else ""
            print(f" Pobór mocy GPU:       {stats['peak_power_w']:.1f} W{p_lim}")
        if stats.get('gpu_util_pct', 0) > 0:
            print(f" Utylizacja GPU:       szczyt: {stats['gpu_util_pct']:.0f}% (średnia: {stats.get('avg_gpu_util_pct', 0):.0f}%)")
        print("=" * 74 + "\n")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        stats = self.stop()
        self.log_session(stats=stats)
        self.print_summary(stats=stats)

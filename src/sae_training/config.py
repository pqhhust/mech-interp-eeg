"""
Configuration for training Sparse Autoencoders on EEG Transformer (REVE) activations.

Adapted from PatchSAE (https://github.com/dynamical-inference/patchsae) for EEG data.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch


class Config:
    """Generic config from dict."""

    def __init__(self, config_dict):
        if not isinstance(config_dict, dict):
            config_dict = config_dict.__dict__
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)


@dataclass
class EEGSAERunnerConfig:
    """
    Configuration for training a sparse autoencoder on REVE EEG transformer activations.
    """

    # ── Model ──────────────────────────────────────────────────────────────
    model_name: str = "brain-bzh/reve-base"
    pos_bank_name: str = "brain-bzh/reve-positions"
    local_model_path: Optional[str] = None
    module_name: str = "resid"          # hook location type: "resid" | "mlp" | "attn"
    block_layer: int = -2               # which transformer block (-1 = last, -2 = second-to-last)
    hook_point: Optional[str] = None    # explicit hook point name (overrides block_layer + module_name)

    # ── EEG Geometry ───────────────────────────────────────────────────────
    # Channel setup (22-ch BCIC2a by default)
    channel_names: List[str] = field(default_factory=lambda: [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP3", "CP1", "CPz", "CP2", "CP4",
        "P1", "Pz", "P2", "POz",
    ])
    n_channels: int = 22
    seq_len: int = 1024                 # number of time samples per trial
    patch_size: int = 200               # REVE patch size
    patch_overlap: int = 20             # REVE patch overlap
    sample_rate: int = 250              # Hz

    # ── SAE Parameters ─────────────────────────────────────────────────────
    d_in: int = 512                     # REVE-base hidden dim
    expansion_factor: int = 16          # SAE dict size = d_in * expansion_factor
    d_sae: Optional[int] = None        # auto-filled in __post_init__
    b_dec_init_method: str = "geometric_median"  # "geometric_median" | "mean" | "zeros"
    gated_sae: bool = False
    class_token: bool = False           # EEG has no class token by default

    # ── Data ───────────────────────────────────────────────────────────────
    dataset_path: str = ""              # path to LMDB dataset
    total_training_tokens: int = 500_000
    n_batches_in_store: int = 32
    store_size: Optional[int] = None
    use_cached_activations: bool = False
    cached_activations_path: Optional[str] = None
    scale_div: float = 100.0            # EEG scaling factor (data / scale_div)

    # ── Training Parameters ────────────────────────────────────────────────
    l1_coefficient: float = 1e-4
    lr: float = 3e-4
    lr_scheduler_name: str = "constantwithwarmup"
    lr_warm_up_steps: int = 500
    batch_size: int = 32                # smaller batches for EEG
    mse_cls_coefficient: float = 1.0

    # ── Resampling / Ghost Grads ───────────────────────────────────────────
    use_ghost_grads: bool = True
    feature_sampling_window: int = 500
    feature_sampling_method: Optional[str] = "anthropic"
    resample_batches: int = 32
    feature_reinit_scale: float = 0.2
    dead_feature_window: int = 250
    dead_feature_estimation_method: str = "no_fire"
    dead_feature_threshold: float = 1e-8

    # ── Weights & Biases ───────────────────────────────────────────────────
    log_to_wandb: bool = False
    wandb_project: str = "eeg-sae"
    wandb_entity: Optional[str] = None
    wandb_log_frequency: int = 10

    # ── Misc ───────────────────────────────────────────────────────────────
    device: str = "cuda"
    seed: int = 42
    dtype: torch.dtype = torch.float32
    n_checkpoints: int = 2
    checkpoint_path: str = "checkpoints"
    run_name: Optional[str] = None

    def __post_init__(self):
        self.store_size = self.n_batches_in_store * self.batch_size
        self.d_sae = self.d_in * self.expansion_factor

        if self.cached_activations_path is None:
            model_tag = self.model_name.replace("/", "_")
            self.cached_activations_path = (
                f"activations/{model_tag}/{self.block_layer}_{self.module_name}"
            )

        if self.run_name is None:
            self.run_name = (
                f"{self.d_sae}-L1-{self.l1_coefficient}-LR-{self.lr}"
                f"-Tokens-{self.total_training_tokens:3.3e}"
            )

        if self.b_dec_init_method not in ["geometric_median", "mean", "zeros"]:
            raise ValueError(f"Invalid b_dec_init_method: {self.b_dec_init_method}")

        self.device = torch.device(self.device)

        # Compute geometry
        step = self.patch_size - self.patch_overlap
        self.n_time_patches = max(1, (self.seq_len - self.patch_size) // step + 1)
        self.n_tokens_per_sample = self.n_channels * self.n_time_patches

        total_training_steps = self.total_training_tokens // self.batch_size
        print(f"[EEG-SAE Config] d_in={self.d_in}, d_sae={self.d_sae}")
        print(f"[EEG-SAE Config] n_channels={self.n_channels}, n_time_patches={self.n_time_patches}")
        print(f"[EEG-SAE Config] Total training steps: {total_training_steps}")
        print(f"[EEG-SAE Config] Run name: {self.run_name}")

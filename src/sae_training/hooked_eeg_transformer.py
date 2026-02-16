"""
Hooked REVE wrapper for SAE training.

Provides a convenient interface for extracting intermediate activations from the
REVE EEG transformer, analogous to HookedVisionTransformer in PatchSAE but
specialized for EEG data and the REVE architecture.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from models.reve import HookedSAEReve, ReveConfig


# Default 22-channel BCIC2a montage
DEFAULT_CHANNELS_22 = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]


class HookedEEGTransformer:
    """Wrapper around HookedSAEReve for convenient activation extraction.

    Handles position resolution, forward passes with hooks, and caching
    of intermediate activations from specific transformer blocks.
    """

    def __init__(
        self,
        model_name: str = "brain-bzh/reve-base",
        pos_bank_name: str = "brain-bzh/reve-positions",
        channel_names: Optional[Sequence[str]] = None,
        local_model_path: Optional[str] = None,
        device: str = "cuda",
    ):
        from transformers import AutoModel

        if channel_names is None:
            channel_names = DEFAULT_CHANNELS_22

        self.channel_names = list(channel_names)
        self.device = device

        # Load REVE model with hooks
        load_from = local_model_path or model_name
        self.model = HookedSAEReve.from_pretrained(
            load_from, trust_remote_code=True, torch_dtype="auto",
        )
        self.model = self.model.to(device)
        self.model.eval()

        # Load position bank
        self.pos_bank = AutoModel.from_pretrained(
            pos_bank_name, trust_remote_code=True, torch_dtype="auto",
        )

        # Resolve channel positions
        positions = self.pos_bank(self.channel_names)  # [C, 3]
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)  # [1, C, 3]
        self.positions = positions.to(device)

        # Model properties
        self.embed_dim = self.model.embed_dim
        self.patch_size = self.model.patch_size
        self.overlap_size = self.model.overlap_size
        self.depth = self.model.config.depth

    def _resolve_block_layer(self, block_layer: int) -> int:
        """Resolve negative block layer indices."""
        if block_layer < 0:
            block_layer = self.depth + block_layer
        return block_layer

    def _get_hook_name(self, block_layer: int, module_name: str) -> str:
        """Get the hook name for a specific block and module."""
        block_layer = self._resolve_block_layer(block_layer)
        hook_map = {
            "resid": f"transformer.layers.{block_layer}.hook_resid_post",
            "resid_pre": f"transformer.layers.{block_layer}.hook_resid_pre",
            "resid_mid": f"transformer.layers.{block_layer}.hook_resid_mid",
            "mlp_in": f"transformer.layers.{block_layer}.ff.hook_mlp_in",
            "mlp_out": f"transformer.layers.{block_layer}.ff.hook_mlp_out",
            "attn_out": f"transformer.layers.{block_layer}.attn.hook_attn_out",
        }
        if module_name not in hook_map:
            raise ValueError(f"Unknown module_name '{module_name}'. Choose from: {list(hook_map.keys())}")
        return hook_map[module_name]

    @torch.no_grad()
    def get_activations(
        self,
        eeg: torch.Tensor,
        block_layer: int,
        module_name: str = "resid",
    ) -> torch.Tensor:
        """Extract activations from a specific layer/module.

        Args:
            eeg: [B, C, T] raw EEG tensor
            block_layer: Transformer block index (supports negative indexing)
            module_name: "resid" | "resid_pre" | "mlp_in" | "mlp_out" | "attn_out"

        Returns:
            activations: [B, N_tokens, d_in] where N_tokens = C * H (channels * time_patches)
        """
        hook_name = self._get_hook_name(block_layer, module_name)
        cache = {}

        def cache_hook(tensor, hook, name=hook_name):
            cache[name] = tensor.detach()
            return tensor

        B = eeg.shape[0]
        pos = self.positions.expand(B, -1, -1).to(eeg.device, dtype=eeg.dtype)

        with torch.amp.autocast(device_type="cuda" if eeg.is_cuda else "cpu", dtype=torch.float16):
            self.model.run_with_hooks(
                eeg, pos,
                fwd_hooks=[(hook_name, cache_hook)],
            )

        activations = cache[hook_name].float()
        return activations

    @torch.no_grad()
    def run_with_cache(
        self,
        eeg: torch.Tensor,
        hook_names: Optional[List[str]] = None,
        block_layer: Optional[int] = None,
        module_name: str = "resid",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run model and return output + cached activations.

        Args:
            eeg: [B, C, T] raw EEG
            hook_names: Explicit list of hook point names. If None, uses block_layer + module_name.
            block_layer: Transformer block index
            module_name: Module type

        Returns:
            output: Model output
            cache: Dict mapping hook names to activation tensors
        """
        if hook_names is None:
            if block_layer is not None:
                hook_names = [self._get_hook_name(block_layer, module_name)]
            else:
                hook_names = []

        cache = {}

        def make_cache_hook(name):
            def hook_fn(tensor, hook):
                cache[name] = tensor.detach()
                return tensor
            return hook_fn

        fwd_hooks = [(name, make_cache_hook(name)) for name in hook_names]

        B = eeg.shape[0]
        pos = self.positions.expand(B, -1, -1).to(eeg.device, dtype=eeg.dtype)

        with torch.amp.autocast(device_type="cuda" if eeg.is_cuda else "cpu", dtype=torch.float16):
            output = self.model.run_with_hooks(
                eeg, pos,
                fwd_hooks=fwd_hooks,
            )

        # Convert cache to float32
        cache = {k: v.float() for k, v in cache.items()}
        return output, cache

    def run_with_hooks(
        self,
        eeg: torch.Tensor,
        fwd_hooks: List[Tuple[str, Callable]],
    ) -> torch.Tensor:
        """Run model with arbitrary forward hooks.

        Args:
            eeg: [B, C, T] raw EEG
            fwd_hooks: List of (hook_name, hook_fn) tuples

        Returns:
            output: Model output
        """
        B = eeg.shape[0]
        pos = self.positions.expand(B, -1, -1).to(eeg.device, dtype=eeg.dtype)

        with torch.amp.autocast(device_type="cuda" if eeg.is_cuda else "cpu", dtype=torch.float16):
            output = self.model.run_with_hooks(
                eeg, pos,
                fwd_hooks=fwd_hooks,
            )
        return output

    def get_all_hook_names(self) -> List[str]:
        """Return all available hook point names."""
        return self.model.get_hook_names()

    def get_layer_hook_names(self, layer_idx: int) -> List[str]:
        """Return hook point names for a specific layer."""
        return self.model.get_layer_hook_names(layer_idx)

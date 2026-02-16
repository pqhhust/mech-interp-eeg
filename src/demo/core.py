"""
Core SAE analysis engine for EEG data.

Provides SAETester — the main class for running SAE on EEG trials, computing
activation distributions, finding top neurons, and generating channel × time
activation masks. This is the EEG analogue of PatchSAE's SAETester.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis.utils import (
    BCIC2A_CLASS_NAMES,
    CHANNEL_POS_2D,
    calculate_class_entropy,
    compute_class_selectivity,
)


class SAETester:
    """Interactive SAE feature analysis for EEG data.

    Analogous to PatchSAE's SAETester, but adapted for EEG:
    - Instead of image patches, works with channel × time-patch tokens
    - Instead of segmentation masks, generates channel × time activation heatmaps
    - Instead of top-activating images, shows top-activating EEG trials
    """

    def __init__(
        self,
        hooked_model,
        sae,
        cfg,
        feature_stats: dict,
        channel_names: List[str],
        n_time_patches: int,
        class_names: Optional[Dict[int, str]] = None,
        noisy_threshold: float = 0.1,
        device: str = "cpu",
    ):
        """
        Args:
            hooked_model: HookedEEGTransformer instance
            sae: Trained SparseAutoencoder
            cfg: Config with block_layer, module_name
            feature_stats: Dict from load_feature_data() with mean_acts, sparsity, per_class_mean_acts, etc.
            channel_names: List of channel names (e.g., BCIC2a 22 channels)
            n_time_patches: Number of time patches per channel
            class_names: Optional mapping from class index to name
            noisy_threshold: Features with mean activation > this are considered noisy
            device: Device string
        """
        self.model = hooked_model
        self.sae = sae
        self.cfg = cfg
        self.stats = feature_stats
        self.channel_names = channel_names
        self.n_channels = len(channel_names)
        self.n_time_patches = n_time_patches
        self.n_tokens = self.n_channels * self.n_time_patches
        self.class_names = class_names or BCIC2A_CLASS_NAMES
        self.noisy_threshold = noisy_threshold
        self.device = device

        # Precompute derived stats
        self.entropy = calculate_class_entropy(self.stats["per_class_mean_acts"])
        self.selectivity, self.preferred_class = compute_class_selectivity(
            self.stats["per_class_mean_acts"]
        )

        # State: current trial
        self._eeg = None
        self._label = None
        self._activations = None
        self._sae_acts = None

    # ── Trial registration ──────────────────────────────────────────────

    def register_trial(self, eeg: np.ndarray, label: Optional[int] = None):
        """Register an EEG trial for analysis.

        Args:
            eeg: [C, T] single EEG trial (numpy or tensor)
            label: Optional ground-truth label
        """
        if isinstance(eeg, np.ndarray):
            eeg = torch.from_numpy(eeg).float()
        if eeg.dim() == 2:
            eeg = eeg.unsqueeze(0)  # [1, C, T]

        self._eeg = eeg.to(self.device)
        self._label = label
        self._activations = None
        self._sae_acts = None

    @property
    def current_eeg(self) -> np.ndarray:
        """Get current EEG as [C, T] numpy array."""
        if self._eeg is None:
            raise RuntimeError("No trial registered. Call register_trial() first.")
        return self._eeg[0].cpu().numpy()

    @property
    def current_label(self) -> Optional[int]:
        return self._label

    # ── Forward pass ────────────────────────────────────────────────────

    def _run_model(self) -> torch.Tensor:
        """Get REVE activations for the registered trial. Cached."""
        if self._activations is None:
            assert self._eeg is not None, "No trial registered"
            self._activations = self.model.get_activations(
                self._eeg, self.cfg.block_layer, self.cfg.module_name
            )
        return self._activations

    def _run_sae(self) -> torch.Tensor:
        """Get SAE feature activations for the registered trial. Cached."""
        if self._sae_acts is None:
            activations = self._run_model()
            self._sae_acts = self.sae.encode(activations.to(self.sae.dtype))
        return self._sae_acts

    # ── Activation analysis ─────────────────────────────────────────────

    def get_activation_distribution(self) -> np.ndarray:
        """Get mean-pooled SAE activation vector for the current trial.

        Returns:
            [d_sae] numpy array of mean activation per SAE feature
        """
        sae_acts = self._run_sae()  # [1, N_tokens, d_sae]
        mean_act = sae_acts[0].mean(dim=0).detach().cpu().numpy()
        return self._filter_noisy(mean_act)

    def get_token_activations(self) -> np.ndarray:
        """Get full token-level SAE activations for the current trial.

        Returns:
            [N_tokens, d_sae] numpy array
        """
        sae_acts = self._run_sae()  # [1, N_tokens, d_sae]
        acts = sae_acts[0].detach().cpu().numpy()
        return acts

    def _filter_noisy(self, features: np.ndarray) -> np.ndarray:
        """Zero out features whose overall mean activation exceeds threshold."""
        noisy_mask = self.stats["mean_acts"].numpy() > self.noisy_threshold
        features = features.copy()
        if features.ndim == 1:
            features[noisy_mask] = 0
        elif features.ndim == 2:
            features[:, noisy_mask] = 0
        return features

    # ── Top neuron finding ──────────────────────────────────────────────

    def get_top_features_trial(self, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Get top-K firing SAE features for the current trial (global).

        Returns:
            (feature_indices, activation_values) both [top_k]
        """
        mean_act = self.get_activation_distribution()
        top_idx = np.argsort(mean_act)[::-1][:top_k]
        return top_idx, mean_act[top_idx]

    def get_top_features_token(
        self, token_idx: int, top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get top-K firing SAE features for a specific token.

        Args:
            token_idx: Token index (0-based). Tokens are ordered as
                       [ch0_t0, ch0_t1, ..., ch0_tH, ch1_t0, ...]
            top_k: Number of top features

        Returns:
            (feature_indices, activation_values) both [top_k]
        """
        token_acts = self.get_token_activations()  # [N_tokens, d_sae]
        tok_act = self._filter_noisy(token_acts[token_idx])
        top_idx = np.argsort(tok_act)[::-1][:top_k]
        return top_idx, tok_act[top_idx]

    def get_top_features_channel(
        self, channel_name: str, top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get top-K features for a specific channel (aggregated over time).

        Args:
            channel_name: e.g., "C3"
            top_k: Number of features

        Returns:
            (feature_indices, activation_values) both [top_k]
        """
        ch_idx = self.channel_names.index(channel_name)
        token_acts = self.get_token_activations()  # [N_tokens, d_sae]
        start = ch_idx * self.n_time_patches
        end = start + self.n_time_patches
        ch_acts = token_acts[start:end].mean(axis=0)  # [d_sae]
        ch_acts = self._filter_noisy(ch_acts)
        top_idx = np.argsort(ch_acts)[::-1][:top_k]
        return top_idx, ch_acts[top_idx]

    # ── Channel × Time mask generation ──────────────────────────────────

    def get_channel_time_mask(self, feature_idx: int) -> np.ndarray:
        """Get channel × time activation mask for a SAE feature.

        This is the EEG analogue of PatchSAE's image segmentation mask.

        Args:
            feature_idx: SAE feature index

        Returns:
            [n_channels, n_time_patches] activation values
        """
        token_acts = self.get_token_activations()  # [N_tokens, d_sae]
        acts = token_acts[:, feature_idx][: self.n_tokens]
        return acts.reshape(self.n_channels, self.n_time_patches)

    def get_channel_activation(
        self, feature_idx: int, aggregation: str = "mean"
    ) -> np.ndarray:
        """Get per-channel activation for a feature (aggregated over time).

        Args:
            feature_idx: SAE feature index
            aggregation: "mean" | "max" | "sum"

        Returns:
            [n_channels] activation values
        """
        mask = self.get_channel_time_mask(feature_idx)
        if aggregation == "mean":
            return mask.mean(axis=1)
        elif aggregation == "max":
            return mask.max(axis=1)
        elif aggregation == "sum":
            return mask.sum(axis=1)
        return mask.mean(axis=1)

    # ── Feature info ────────────────────────────────────────────────────

    def get_feature_info(self, feature_idx: int) -> dict:
        """Get summary statistics for a SAE feature.

        Returns dict with: mean_act, sparsity, entropy, selectivity,
                          preferred_class, preferred_class_name
        """
        return {
            "feature_idx": feature_idx,
            "mean_act": float(self.stats["mean_acts"][feature_idx]),
            "sparsity": float(self.stats["sparsity"][feature_idx]),
            "entropy": float(self.entropy[feature_idx]),
            "selectivity": float(self.selectivity[feature_idx]),
            "preferred_class": int(self.preferred_class[feature_idx]),
            "preferred_class_name": self.class_names.get(
                int(self.preferred_class[feature_idx]), "Unknown"
            ),
            "per_class_acts": self.stats["per_class_mean_acts"][:, feature_idx]
            .numpy()
            .tolist(),
        }

    # ── Reconstruction analysis ─────────────────────────────────────────

    def get_reconstruction_error(self) -> dict:
        """Compute reconstruction metrics for the current trial.

        Returns:
            Dict with mse, relative_mse, cosine_sim, L0
        """
        activations = self._run_model()  # [1, N_tokens, d_in]
        sae_out, feature_acts, loss_dict = self.sae(activations.to(self.sae.dtype))

        x = activations.float()
        x_hat = sae_out.float()

        mse = torch.pow(x - x_hat, 2).mean().item()
        relative_mse = (
            mse / torch.pow(x, 2).mean().item() if torch.pow(x, 2).mean().item() > 0 else 0
        )

        # Cosine similarity
        cos_sim = (
            torch.nn.functional.cosine_similarity(
                x.reshape(-1, x.shape[-1]),
                x_hat.reshape(-1, x_hat.shape[-1]),
                dim=-1,
            )
            .mean()
            .item()
        )

        # L0: average number of active features per token
        l0 = (feature_acts > 0).float().sum(dim=-1).mean().item()

        return {
            "mse": mse,
            "relative_mse": relative_mse,
            "cosine_similarity": cos_sim,
            "L0": l0,
        }

    # ── Batch analysis helpers ──────────────────────────────────────────

    def analyze_batch(
        self,
        eeg_batch: torch.Tensor,
        labels_batch: torch.Tensor,
        top_k: int = 5,
    ) -> List[dict]:
        """Analyze a batch of EEG trials.

        Args:
            eeg_batch: [B, C, T]
            labels_batch: [B]
            top_k: Number of top features per trial

        Returns:
            List of summary dicts per trial
        """
        results = []
        for i in range(eeg_batch.shape[0]):
            self.register_trial(eeg_batch[i].cpu().numpy(), int(labels_batch[i]))
            top_feats, top_vals = self.get_top_features_trial(top_k)
            recon = self.get_reconstruction_error()
            results.append({
                "trial_idx": i,
                "label": int(labels_batch[i]),
                "label_name": self.class_names.get(int(labels_batch[i]), "?"),
                "top_features": top_feats.tolist(),
                "top_activations": top_vals.tolist(),
                "reconstruction": recon,
            })
        return results

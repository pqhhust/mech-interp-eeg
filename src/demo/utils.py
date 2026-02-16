"""
Loading utilities for the EEG-SAE demo app.

Provides one-call initialization of SAETester with model, SAE, and stats.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from analysis.utils import BCIC2A_CLASS_NAMES, load_feature_data
from src.demo.core import SAETester
from src.sae_training.config import EEGSAERunnerConfig, Config
from src.sae_training.hooked_eeg_transformer import HookedEEGTransformer
from src.sae_training.sparse_autoencoder import SparseAutoencoder
from tasks.utils import BCIC2A_CHANNELS


def load_sae_tester(
    sae_path: str,
    feature_data_path: str,
    model_name: str = "brain-bzh/reve-base",
    pos_bank_name: str = "brain-bzh/reve-positions",
    local_model_path: Optional[str] = None,
    channel_names: Optional[List[str]] = None,
    block_layer: int = -2,
    module_name: str = "resid",
    n_time_patches: int = 5,
    noisy_threshold: float = 0.1,
    device: str = "cpu",
) -> SAETester:
    """Initialize SAETester with all dependencies.

    Args:
        sae_path: Path to trained SAE checkpoint (.pt)
        feature_data_path: Path to precomputed feature data (.npz)
        model_name: REVE HuggingFace model name
        pos_bank_name: REVE position bank name
        local_model_path: Optional local model path
        channel_names: EEG channel names (default: BCIC2a 22-ch)
        block_layer: Transformer block to extract from
        module_name: Hook type
        n_time_patches: Number of time patches
        noisy_threshold: Threshold for noisy feature filtering
        device: Device

    Returns:
        SAETester instance ready for analysis
    """
    if channel_names is None:
        channel_names = BCIC2A_CHANNELS

    print(f"Loading REVE model: {model_name}...")
    model = HookedEEGTransformer(
        model_name=model_name,
        pos_bank_name=pos_bank_name,
        channel_names=channel_names,
        local_model_path=local_model_path,
        device=device,
    )

    print(f"Loading SAE from: {sae_path}...")
    sae = SparseAutoencoder.load_from_pretrained(sae_path, device=device)
    sae.eval()

    print(f"Loading feature stats from: {feature_data_path}...")
    stats = load_feature_data(feature_data_path, device=device)

    # Create a lightweight config object
    cfg = Config({
        "block_layer": block_layer,
        "module_name": module_name,
    })

    tester = SAETester(
        hooked_model=model,
        sae=sae,
        cfg=cfg,
        feature_stats=stats,
        channel_names=channel_names,
        n_time_patches=n_time_patches,
        noisy_threshold=noisy_threshold,
        device=device,
    )

    print(f"SAETester initialized. d_sae={sae.d_sae}, {len(channel_names)} channels, {n_time_patches} time patches")
    return tester


def load_dataset_for_demo(
    db_path: str,
    batch_size: int = 32,
    test_subject_ids: Optional[List[int]] = None,
    scale_div: float = 100.0,
) -> dict:
    """Load BCIC2a dataset for the demo.

    Returns dict with 'train' and 'test' dataloaders.
    """
    from downstream.bcic2a_dataset import get_bcic2a_dataloaders

    if test_subject_ids is None:
        test_subject_ids = [9]

    loaders = get_bcic2a_dataloaders(
        db_path=db_path,
        batch_size=batch_size,
        test_subject_ids=test_subject_ids,
        scale_div=scale_div,
    )
    return loaders

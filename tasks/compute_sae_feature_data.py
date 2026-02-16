"""
Compute per-feature statistics from trained SAE on EEG data.

For each SAE feature, computes:
  - Mean activation
  - Sparsity (fraction of tokens where it fires)
  - Top-activating trials (indices + values)
  - Per-class activation statistics

This is the EEG analogue of PatchSAE's compute_sae_feature_data task.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from downstream.bcic2a_dataset import get_bcic2a_dataloaders
from src.sae_training.hooked_eeg_transformer import HookedEEGTransformer
from src.sae_training.sparse_autoencoder import SparseAutoencoder
from tasks.utils import BCIC2A_CHANNELS, parse_channel_names


@torch.no_grad()
def compute_feature_data(
    model: HookedEEGTransformer,
    sae: SparseAutoencoder,
    dataloader: torch.utils.data.DataLoader,
    block_layer: int,
    module_name: str = "resid",
    device: str = "cuda",
    num_top_trials: int = 20,
) -> dict:
    """Compute SAE feature statistics over a dataset.

    Returns:
        Dictionary with:
        - feature_mean_acts: [d_sae] mean activation per feature
        - feature_sparsity: [d_sae] fraction of tokens where feature fires
        - top_activating_trials: [d_sae, num_top_trials] indices of top trials
        - per_class_mean_acts: [n_classes, d_sae] per-class mean activation
    """
    d_sae = sae.d_sae
    sae.eval()

    # Accumulators
    total_mean_acts = np.zeros(d_sae)
    total_sparsity = np.zeros(d_sae)
    total_tokens = 0
    trial_idx = 0

    top_values = np.full((d_sae, num_top_trials), -np.inf)
    top_indices = np.zeros((d_sae, num_top_trials), dtype=np.int64)

    class_sums = defaultdict(lambda: np.zeros(d_sae))
    class_counts = defaultdict(int)

    for batch in tqdm(dataloader, desc="Computing feature data"):
        eeg, labels = batch
        eeg = eeg.to(device)
        B = eeg.shape[0]

        activations = model.get_activations(eeg, block_layer, module_name)
        feature_acts = sae.encode(activations.to(sae.dtype))  # [B, N, d_sae]

        # Pool over tokens for trial-level stats
        pooled = feature_acts.mean(dim=1).cpu().numpy()  # [B, d_sae]

        # Update mean and sparsity
        total_mean_acts += pooled.sum(axis=0)
        token_sparsity = (feature_acts > 0).float().sum(dim=1).cpu().numpy()  # [B, d_sae]
        total_sparsity += token_sparsity.sum(axis=0)
        total_tokens += B * feature_acts.shape[1]

        # Update top-K trials
        for i in range(B):
            global_idx = trial_idx + i
            for f in range(d_sae):
                val = pooled[i, f]
                min_idx = top_values[f].argmin()
                if val > top_values[f, min_idx]:
                    top_values[f, min_idx] = val
                    top_indices[f, min_idx] = global_idx

        # Per-class stats
        for i, label in enumerate(labels.numpy()):
            class_sums[int(label)] += pooled[i]
            class_counts[int(label)] += 1

        trial_idx += B

    # Finalize
    n_trials = trial_idx
    feature_mean_acts = total_mean_acts / max(n_trials, 1)
    feature_sparsity = total_sparsity / max(total_tokens, 1)

    # Sort top trials
    for f in range(d_sae):
        sort_idx = np.argsort(top_values[f])[::-1]
        top_values[f] = top_values[f, sort_idx]
        top_indices[f] = top_indices[f, sort_idx]

    # Per-class means
    n_classes = max(class_counts.keys()) + 1 if class_counts else 0
    per_class_mean = np.zeros((n_classes, d_sae))
    for cls_idx in range(n_classes):
        if class_counts[cls_idx] > 0:
            per_class_mean[cls_idx] = class_sums[cls_idx] / class_counts[cls_idx]

    return {
        "feature_mean_acts": feature_mean_acts,
        "feature_sparsity": feature_sparsity,
        "top_activating_trial_values": top_values,
        "top_activating_trial_indices": top_indices,
        "per_class_mean_acts": per_class_mean,
        "n_trials": n_trials,
        "n_classes": n_classes,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute SAE feature statistics on EEG data")

    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument("--sae_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="brain-bzh/reve-base")
    parser.add_argument("--pos_bank_name", type=str, default="brain-bzh/reve-positions")
    parser.add_argument("--local_model_path", type=str, default=None)
    parser.add_argument("--block_layer", type=int, default=-2)
    parser.add_argument("--module_name", type=str, default="resid")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--scale_div", type=float, default=100.0)
    parser.add_argument("--num_top_trials", type=int, default=20)
    parser.add_argument("--channel_names", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/feature_data")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    channels = parse_channel_names(args.channel_names) or BCIC2A_CHANNELS
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...")
    model = HookedEEGTransformer(
        model_name=args.model_name,
        pos_bank_name=args.pos_bank_name,
        channel_names=channels,
        local_model_path=args.local_model_path,
        device=args.device,
    )

    print(f"Loading SAE from {args.sae_path}...")
    sae = SparseAutoencoder.load_from_pretrained(args.sae_path, device=args.device)
    sae.eval()

    loaders = get_bcic2a_dataloaders(
        db_path=args.db_path,
        batch_size=args.batch_size,
        test_subject_ids=[9],
        scale_div=args.scale_div,
    )

    print("Computing feature data on training set...")
    data = compute_feature_data(
        model, sae, loaders["train"],
        args.block_layer, args.module_name,
        args.device, args.num_top_trials,
    )

    # Save
    output_path = Path(args.output_dir) / "sae_feature_data.npz"
    np.savez(
        output_path,
        feature_mean_acts=data["feature_mean_acts"],
        feature_sparsity=data["feature_sparsity"],
        top_activating_trial_values=data["top_activating_trial_values"],
        top_activating_trial_indices=data["top_activating_trial_indices"],
        per_class_mean_acts=data["per_class_mean_acts"],
    )
    print(f"Feature data saved to {output_path}")

    # Print summary
    print(f"\n=== Feature Statistics ===")
    print(f"Total features: {sae.d_sae}")
    print(f"Mean sparsity: {data['feature_sparsity'].mean():.6f}")
    print(f"Dead features (sparsity < 1e-6): {(data['feature_sparsity'] < 1e-6).sum()}")
    print(f"Number of classes: {data['n_classes']}")
    print(f"Total trials: {data['n_trials']}")


if __name__ == "__main__":
    main()

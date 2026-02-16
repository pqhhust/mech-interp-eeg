"""
Linear probe evaluation of SAE features on BCIC2a (4-class motor imagery).

This script:
  1) Loads a pretrained REVE model + trained SAE
  2) Extracts SAE features for each EEG trial in train/test
  3) Trains a linear probe (logistic regression or small MLP) on the SAE features
  4) Reports classification accuracy, kappa, and F1

This is the EEG analogue of PatchSAE's classification_with_top_k_masking task.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from tqdm import tqdm

from downstream.bcic2a_dataset import get_bcic2a_dataloaders
from src.sae_training.hooked_eeg_transformer import HookedEEGTransformer
from src.sae_training.sparse_autoencoder import SparseAutoencoder


# ── Feature extraction ──────────────────────────────────────────────────────


@torch.no_grad()
def extract_sae_features(
    model: HookedEEGTransformer,
    sae: SparseAutoencoder,
    dataloader: torch.utils.data.DataLoader,
    block_layer: int,
    module_name: str = "resid",
    device: str = "cuda",
    pooling: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract SAE features for all samples in a dataloader.

    For each trial:
      1) Get REVE activations [B, N_tokens, d_in]
      2) Encode with SAE → [B, N_tokens, d_sae]
      3) Pool over tokens → [B, d_sae]

    Args:
        model: HookedEEGTransformer
        sae: Trained SparseAutoencoder
        dataloader: DataLoader yielding (eeg, labels)
        block_layer: Which transformer block to extract from
        module_name: Hook type
        device: Device
        pooling: "mean" | "max" | "meanmax" | "binary"

    Returns:
        features: [N, d_feat] numpy array
        labels: [N] numpy array
    """
    sae.eval()
    all_features, all_labels = [], []

    for batch in tqdm(dataloader, desc="Extracting SAE features"):
        eeg, labels = batch
        eeg = eeg.to(device)

        # Get model activations
        activations = model.get_activations(eeg, block_layer, module_name)

        # Encode with SAE
        feature_acts = sae.encode(activations.to(sae.dtype))  # [B, N_tokens, d_sae]

        # Pool over tokens
        if pooling == "mean":
            pooled = feature_acts.mean(dim=1)           # [B, d_sae]
        elif pooling == "max":
            pooled = feature_acts.amax(dim=1)            # [B, d_sae]
        elif pooling == "meanmax":
            m = feature_acts.mean(dim=1)
            M = feature_acts.amax(dim=1)
            pooled = torch.cat([m, M], dim=-1)           # [B, 2*d_sae]
        elif pooling == "binary":
            pooled = (feature_acts > 0).float().mean(dim=1)  # [B, d_sae]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        all_features.append(pooled.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return features, labels


# ── Raw activation baseline ─────────────────────────────────────────────────


@torch.no_grad()
def extract_raw_features(
    model: HookedEEGTransformer,
    dataloader: torch.utils.data.DataLoader,
    block_layer: int,
    module_name: str = "resid",
    device: str = "cuda",
    pooling: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract raw (non-SAE) REVE features for comparison."""
    all_features, all_labels = [], []

    for batch in tqdm(dataloader, desc="Extracting raw features"):
        eeg, labels = batch
        eeg = eeg.to(device)
        activations = model.get_activations(eeg, block_layer, module_name)

        if pooling == "mean":
            pooled = activations.mean(dim=1)
        elif pooling == "max":
            pooled = activations.amax(dim=1)
        else:
            pooled = activations.mean(dim=1)

        all_features.append(pooled.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


# ── Linear probe ────────────────────────────────────────────────────────────


def linear_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    C: float = 1.0,
    max_iter: int = 2000,
) -> Dict[str, float]:
    """Train and evaluate a linear probe (logistic regression).

    Args:
        train_features: [N_train, d_feat]
        train_labels: [N_train]
        test_features: [N_test, d_feat]
        test_labels: [N_test]
        C: Regularization strength (inverse)
        max_iter: Max iterations

    Returns:
        Dictionary of metrics
    """
    clf = LogisticRegression(
        C=C, max_iter=max_iter, solver="lbfgs",
        multi_class="multinomial", n_jobs=-1,
    )
    clf.fit(train_features, train_labels)
    preds = clf.predict(test_features)

    metrics = {
        "accuracy": accuracy_score(test_labels, preds),
        "balanced_accuracy": balanced_accuracy_score(test_labels, preds),
        "cohen_kappa": cohen_kappa_score(test_labels, preds),
        "f1_weighted": f1_score(test_labels, preds, average="weighted"),
        "f1_macro": f1_score(test_labels, preds, average="macro"),
    }
    return metrics


# ── Top-K SAE feature masking evaluation ────────────────────────────────────


@torch.no_grad()
def topk_sae_feature_probe(
    model: HookedEEGTransformer,
    sae: SparseAutoencoder,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    block_layer: int,
    module_name: str = "resid",
    device: str = "cuda",
    topk_list: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate classification with only top-K SAE features per class.

    1) Extract full SAE features for train set
    2) Compute per-class mean feature activation
    3) Select top-K features for each class
    4) Re-evaluate with only those features active

    Args:
        model, sae, train_loader, test_loader: Standard components
        block_layer, module_name, device: Model config
        topk_list: List of K values to try

    Returns:
        Dict mapping "topk_{k}" to metric dicts
    """
    if topk_list is None:
        topk_list = [10, 50, 100, 500, 1000]

    d_sae = sae.d_sae

    # Step 1: Get per-class feature activations
    print("Computing per-class SAE feature activations...")
    class_feature_sums = defaultdict(lambda: np.zeros(d_sae))
    class_counts = defaultdict(int)

    for batch in tqdm(train_loader, desc="Scanning train set"):
        eeg, labels = batch
        eeg = eeg.to(device)
        activations = model.get_activations(eeg, block_layer, module_name)
        feature_acts = sae.encode(activations.to(sae.dtype))
        pooled = feature_acts.mean(dim=1).cpu().numpy()

        for i, label in enumerate(labels.numpy()):
            class_feature_sums[label] += pooled[i]
            class_counts[label] += 1

    # Mean activation per class
    n_classes = len(class_feature_sums)
    class_means = np.zeros((n_classes, d_sae))
    for cls_idx in range(n_classes):
        if class_counts[cls_idx] > 0:
            class_means[cls_idx] = class_feature_sums[cls_idx] / class_counts[cls_idx]

    # Step 2: For each K, select top features and evaluate
    results = {}
    for k in topk_list:
        if k > d_sae:
            k = d_sae

        # Union of top-K features across all classes
        top_features = set()
        for cls_idx in range(n_classes):
            top_k_indices = np.argsort(class_means[cls_idx])[-k:]
            top_features.update(top_k_indices.tolist())

        feature_mask = np.zeros(d_sae, dtype=bool)
        feature_mask[list(top_features)] = True

        # Extract masked features for train and test
        train_feats, train_labels = _extract_masked_sae_features(
            model, sae, train_loader, block_layer, module_name, device, feature_mask,
        )
        test_feats, test_labels = _extract_masked_sae_features(
            model, sae, test_loader, block_layer, module_name, device, feature_mask,
        )

        metrics = linear_probe(train_feats, train_labels, test_feats, test_labels)
        results[f"topk_{k}"] = metrics
        print(f"  Top-{k} ({len(top_features)} unique features): acc={metrics['accuracy']:.4f}")

    return results


@torch.no_grad()
def _extract_masked_sae_features(
    model, sae, dataloader, block_layer, module_name, device, feature_mask,
):
    """Extract SAE features with a binary mask applied."""
    all_features, all_labels = [], []
    mask_tensor = torch.from_numpy(feature_mask).float().to(device)

    for batch in dataloader:
        eeg, labels = batch
        eeg = eeg.to(device)
        activations = model.get_activations(eeg, block_layer, module_name)
        feature_acts = sae.encode(activations.to(sae.dtype))
        feature_acts = feature_acts * mask_tensor  # mask out non-top features
        pooled = feature_acts.mean(dim=1).cpu().numpy()
        all_features.append(pooled)
        all_labels.append(labels.numpy())

    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="BCIC2a linear probe with SAE features")

    # Data
    parser.add_argument("--db_path", type=str, required=True, help="Path to BCIC2a LMDB")
    parser.add_argument("--test_subject", type=int, default=9, help="Test subject ID (1-based)")
    parser.add_argument("--val_subject", type=int, default=None, help="Val subject ID (1-based, optional)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--scale_div", type=float, default=100.0)

    # Model
    parser.add_argument("--model_name", type=str, default="brain-bzh/reve-base")
    parser.add_argument("--pos_bank_name", type=str, default="brain-bzh/reve-positions")
    parser.add_argument("--local_model_path", type=str, default=None)
    parser.add_argument("--block_layer", type=int, default=-2)
    parser.add_argument("--module_name", type=str, default="resid")

    # SAE
    parser.add_argument("--sae_path", type=str, required=True, help="Path to trained SAE checkpoint")

    # Evaluation
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "meanmax", "binary"])
    parser.add_argument("--regularization", type=float, default=1.0, help="LogisticRegression C parameter")
    parser.add_argument("--run_topk", action="store_true", help="Also run top-K feature masking evaluation")
    parser.add_argument("--run_raw_baseline", action="store_true", help="Also run raw feature baseline")
    parser.add_argument("--topk_list", type=str, default="10,50,100,500,1000",
                        help="Comma-separated list of K values for top-K eval")

    # Output
    parser.add_argument("--output_dir", type=str, default="results/bcic2a")
    parser.add_argument("--device", type=str, default="cuda")

    # Channels
    parser.add_argument("--channel_names", type=str, default=None,
                        help="Comma-separated channel names (defaults to BCIC2a 22-ch)")

    args = parser.parse_args()

    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse channels
    if args.channel_names:
        channel_names = [ch.strip() for ch in args.channel_names.split(",")]
    else:
        channel_names = None  # use default 22-ch

    # ── Load model ──
    print("Loading REVE model...")
    model = HookedEEGTransformer(
        model_name=args.model_name,
        pos_bank_name=args.pos_bank_name,
        channel_names=channel_names,
        local_model_path=args.local_model_path,
        device=device,
    )

    # ── Load SAE ──
    print(f"Loading SAE from {args.sae_path}...")
    sae = SparseAutoencoder.load_from_pretrained(args.sae_path, device=device)
    sae.eval()

    # ── Load data ──
    test_ids = [args.test_subject]
    val_ids = [args.val_subject] if args.val_subject else None

    loaders = get_bcic2a_dataloaders(
        db_path=args.db_path,
        batch_size=args.batch_size,
        test_subject_ids=test_ids,
        val_subject_ids=val_ids,
        scale_div=args.scale_div,
    )

    results = {}

    # ── SAE feature probe ──
    print("\n=== SAE Feature Linear Probe ===")
    train_feats, train_labels = extract_sae_features(
        model, sae, loaders["train"], args.block_layer, args.module_name,
        device, args.pooling,
    )
    test_feats, test_labels = extract_sae_features(
        model, sae, loaders["test"], args.block_layer, args.module_name,
        device, args.pooling,
    )
    sae_metrics = linear_probe(train_feats, train_labels, test_feats, test_labels, C=args.regularization)
    results["sae_probe"] = sae_metrics
    print(f"SAE probe results: {json.dumps(sae_metrics, indent=2)}")

    # ── Raw baseline ──
    if args.run_raw_baseline:
        print("\n=== Raw Feature Linear Probe (baseline) ===")
        raw_train, raw_train_labels = extract_raw_features(
            model, loaders["train"], args.block_layer, args.module_name, device, args.pooling,
        )
        raw_test, raw_test_labels = extract_raw_features(
            model, loaders["test"], args.block_layer, args.module_name, device, args.pooling,
        )
        raw_metrics = linear_probe(raw_train, raw_train_labels, raw_test, raw_test_labels, C=args.regularization)
        results["raw_probe"] = raw_metrics
        print(f"Raw probe results: {json.dumps(raw_metrics, indent=2)}")

    # ── Top-K masking ──
    if args.run_topk:
        print("\n=== Top-K SAE Feature Masking ===")
        topk_list = [int(k) for k in args.topk_list.split(",")]
        topk_results = topk_sae_feature_probe(
            model, sae, loaders["train"], loaders["test"],
            args.block_layer, args.module_name, device, topk_list,
        )
        results["topk"] = topk_results

    # ── Save results ──
    output_path = Path(args.output_dir) / f"bcic2a_subject{args.test_subject}_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

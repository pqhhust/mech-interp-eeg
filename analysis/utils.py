"""
Analysis utilities for EEG-SAE feature interpretation.

Provides tools for loading SAE feature statistics, computing entropy,
generating scatter plots, and visualizing EEG-specific activation patterns
(channel topomaps, temporal activation heatmaps, per-class comparisons).
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

# ── BCIC2a 22-channel standard 10-20 layout ─────────────────────────────
# Approximate (x, y) positions for 2D topomap visualization.
# x: left(-1)→right(+1), y: back(-1)→front(+1)
CHANNEL_POS_2D = {
    "Fz":  ( 0.000,  0.700),
    "FC3": (-0.400,  0.450),
    "FC1": (-0.150,  0.450),
    "FCz": ( 0.000,  0.450),
    "FC2": ( 0.150,  0.450),
    "FC4": ( 0.400,  0.450),
    "C5":  (-0.700,  0.200),
    "C3":  (-0.400,  0.200),
    "C1":  (-0.150,  0.200),
    "Cz":  ( 0.000,  0.200),
    "C2":  ( 0.150,  0.200),
    "C4":  ( 0.400,  0.200),
    "C6":  ( 0.700,  0.200),
    "CP3": (-0.400, -0.050),
    "CP1": (-0.150, -0.050),
    "CPz": ( 0.000, -0.050),
    "CP2": ( 0.150, -0.050),
    "CP4": ( 0.400, -0.050),
    "P1":  (-0.150, -0.300),
    "Pz":  ( 0.000, -0.300),
    "P2":  ( 0.150, -0.300),
    "POz": ( 0.000, -0.550),
}

BCIC2A_CLASS_NAMES = {0: "Left Hand", 1: "Right Hand", 2: "Both Feet", 3: "Tongue"}


# ── Loading helpers ─────────────────────────────────────────────────────────


def load_feature_data(path: str, device: str = "cpu") -> dict:
    """Load precomputed SAE feature statistics from .npz file.

    Returns dict with keys:
        mean_acts, sparsity, top_values, top_indices, per_class_mean_acts
    """
    data = np.load(path, allow_pickle=True)
    stats = {
        "mean_acts": torch.from_numpy(data["feature_mean_acts"]).to(device),
        "sparsity": torch.from_numpy(data["feature_sparsity"]).to(device),
        "top_values": torch.from_numpy(data["top_activating_trial_values"]).to(device),
        "top_indices": torch.from_numpy(data["top_activating_trial_indices"]).long().to(device),
        "per_class_mean_acts": torch.from_numpy(data["per_class_mean_acts"]).to(device),
    }
    print(f"Loaded feature statistics from {path}")
    print(f"  d_sae = {stats['mean_acts'].shape[0]}")
    print(f"  n_classes = {stats['per_class_mean_acts'].shape[0]}")
    return stats



def calculate_class_entropy(
    per_class_mean_acts: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Calculate class-distribution entropy for each SAE feature.

    A feature with low entropy is class-specific; high entropy means it fires
    uniformly across classes.

    Args:
        per_class_mean_acts: [n_classes, d_sae]
        eps: Numerical stability

    Returns:
        entropy: [d_sae]
    """
    # Normalize to probabilities per feature
    sums = per_class_mean_acts.sum(dim=0, keepdim=True).clamp(min=eps)  # [1, d_sae]
    probs = per_class_mean_acts / sums  # [n_classes, d_sae]
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=0)  # [d_sae]
    return entropy


def compute_class_selectivity(
    per_class_mean_acts: torch.Tensor,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute class selectivity index for each feature.

    Selectivity = (max_class - mean_others) / (max_class + mean_others + eps)
    Ranges from ~0 (non-selective) to ~1 (very class-specific).

    Args:
        per_class_mean_acts: [n_classes, d_sae]

    Returns:
        selectivity: [d_sae]
        preferred_class: [d_sae] (index of highest-activating class)
    """
    n_classes = per_class_mean_acts.shape[0]
    max_vals, preferred = per_class_mean_acts.max(dim=0)  # [d_sae]

    # Mean of other classes
    total = per_class_mean_acts.sum(dim=0)
    mean_others = (total - max_vals) / max(n_classes - 1, 1)

    selectivity = (max_vals - mean_others) / (max_vals + mean_others + eps)
    return selectivity, preferred


# ── Scatter / overview plots ────────────────────────────────────────────────


def feature_overview_scatter(
    stats: dict,
    mask: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    eps: float = 1e-9,
):
    """Interactive Plotly scatter: sparsity vs mean activation, colored by entropy.

    Args:
        stats: Dict from load_feature_data()
        mask: Optional boolean mask [d_sae] to subset features
        save_path: If given, save as HTML
    """
    per_class = stats["per_class_mean_acts"]
    entropy = calculate_class_entropy(per_class)

    if mask is None:
        mask = torch.ones_like(stats["sparsity"], dtype=torch.bool)

    indices = torch.where(mask)[0]
    selectivity, preferred = compute_class_selectivity(per_class)

    df = pd.DataFrame({
        "log10_sparsity": torch.log10(stats["sparsity"][mask] + eps).numpy(),
        "log10_mean_acts": torch.log10(stats["mean_acts"][mask] + eps).numpy(),
        "entropy": entropy[mask].numpy(),
        "selectivity": selectivity[mask].numpy(),
        "preferred_class": preferred[mask].numpy(),
        "index": indices.numpy(),
    })

    fig = px.scatter(
        df,
        x="log10_sparsity",
        y="log10_mean_acts",
        color="entropy",
        hover_data=["index", "selectivity", "preferred_class"],
        marginal_x="histogram",
        marginal_y="histogram",
        opacity=0.5,
        title="SAE Feature Overview: sparsity vs activation (colored by class entropy)",
        labels={
            "log10_sparsity": "log₁₀(sparsity)",
            "log10_mean_acts": "log₁₀(mean activation)",
            "entropy": "Class entropy",
        },
    )
    fig.update_layout(template="plotly_white")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.write_html(save_path)
        print(f"Scatter plot saved to {save_path}")

    return fig


def dead_feature_summary(stats: dict, threshold: float = 1e-6) -> dict:
    """Summarize dead and alive features.

    Returns:
        Dict with counts and indices.
    """
    sparsity = stats["sparsity"]
    dead_mask = sparsity < threshold
    alive_mask = ~dead_mask

    return {
        "n_total": int(sparsity.shape[0]),
        "n_dead": int(dead_mask.sum().item()),
        "n_alive": int(alive_mask.sum().item()),
        "frac_dead": float(dead_mask.float().mean().item()),
        "dead_indices": torch.where(dead_mask)[0].tolist(),
        "alive_mask": alive_mask,
    }


# ── Channel × Time activation heatmap ──────────────────────────────────────


def plot_channel_time_activation(
    sae_feature_acts: np.ndarray,
    feature_idx: int,
    channel_names: List[str],
    n_time_patches: int,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Plot a channel × time heatmap of SAE feature activation.

    Shows how a specific SAE feature activates across channels and time patches,
    analogous to how PatchSAE shows feature activation overlaid on image patches.

    Args:
        sae_feature_acts: [N_tokens, d_sae] or [n_channels * n_time_patches, d_sae]
        feature_idx: Which SAE feature to visualize
        channel_names: List of channel names (length = n_channels)
        n_time_patches: Number of time patches per channel
        title: Plot title
        save_path: Optional save path for the figure
    """
    n_channels = len(channel_names)
    acts = sae_feature_acts[:, feature_idx]  # [n_channels * n_time_patches]

    # Reshape to [n_channels, n_time_patches]
    if len(acts) >= n_channels * n_time_patches:
        acts = acts[:n_channels * n_time_patches]
    heatmap = acts.reshape(n_channels, n_time_patches)

    fig, ax = plt.subplots(1, 1, figsize=(max(8, n_time_patches * 0.8), max(6, n_channels * 0.35)))
    im = ax.imshow(heatmap, aspect="auto", cmap="RdYlBu_r", interpolation="nearest")
    ax.set_yticks(range(n_channels))
    ax.set_yticklabels(channel_names, fontsize=9)
    ax.set_xlabel("Time patch index", fontsize=11)
    ax.set_ylabel("Channel", fontsize=11)
    ax.set_xticks(range(n_time_patches))
    plt.colorbar(im, ax=ax, label="Feature activation")
    ax.set_title(title or f"SAE Feature {feature_idx}: Channel × Time Activation", fontsize=13)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved to {save_path}")

    return fig


def plot_channel_time_activation_plotly(
    sae_feature_acts: np.ndarray,
    feature_idx: int,
    channel_names: List[str],
    n_time_patches: int,
    title: Optional[str] = None,
) -> go.Figure:
    """Interactive Plotly heatmap of SAE feature activation on channel × time grid."""
    n_channels = len(channel_names)
    acts = sae_feature_acts[:, feature_idx]
    if len(acts) >= n_channels * n_time_patches:
        acts = acts[:n_channels * n_time_patches]
    heatmap = acts.reshape(n_channels, n_time_patches)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap,
        x=[f"T{i}" for i in range(n_time_patches)],
        y=channel_names,
        colorscale="RdYlBu_r",
        colorbar=dict(title="Activation"),
    ))
    fig.update_layout(
        title=title or f"SAE Feature {feature_idx}: Channel × Time",
        xaxis_title="Time patch",
        yaxis_title="Channel",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        height=max(400, n_channels * 25),
    )
    return fig


# ── Topomap visualization ──────────────────────────────────────────────────


def plot_topomap(
    values: np.ndarray,
    channel_names: List[str],
    title: str = "Channel Activation",
    channel_pos: Optional[Dict[str, Tuple[float, float]]] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "RdYlBu_r",
    show_labels: bool = True,
) -> plt.Figure:
    """Plot a scalp topomap showing activation per channel.

    Uses standard 10-20 positions for BCIC2a channels. Each channel is shown
    as a colored circle on the scalp layout.

    Args:
        values: [n_channels] activation values (one per channel)
        channel_names: List of channel names
        title: Plot title
        channel_pos: Optional dict mapping channel name → (x, y)
        ax: Optional matplotlib axes
        cmap: Colormap name
        show_labels: Whether to show channel name labels
    """
    if channel_pos is None:
        channel_pos = CHANNEL_POS_2D

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.figure

    xs, ys, vals, labels = [], [], [], []
    for ch, val in zip(channel_names, values):
        if ch in channel_pos:
            x, y = channel_pos[ch]
            xs.append(x)
            ys.append(y)
            vals.append(val)
            labels.append(ch)

    xs, ys, vals = np.array(xs), np.array(ys), np.array(vals)

    # Head outline
    head = plt.Circle((0, 0.2), 0.75, fill=False, linewidth=2, color="black")
    ax.add_patch(head)
    # Nose
    ax.plot([-0.06, 0, 0.06], [0.93, 1.0, 0.93], "k-", linewidth=2)
    # Ears
    ax.plot([-0.78, -0.83, -0.78], [0.35, 0.2, 0.05], "k-", linewidth=2)
    ax.plot([0.78, 0.83, 0.78], [0.35, 0.2, 0.05], "k-", linewidth=2)

    # Scatter with colormap
    vmin, vmax = vals.min(), vals.max()
    if vmin == vmax:
        vmax = vmin + 1e-6
    scatter = ax.scatter(xs, ys, c=vals, cmap=cmap, s=350, edgecolors="black",
                         linewidths=1.5, vmin=vmin, vmax=vmax, zorder=3)

    if show_labels:
        for x, y, label in zip(xs, ys, labels):
            ax.annotate(label, (x, y), ha="center", va="center", fontsize=7,
                        fontweight="bold", zorder=4)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.7, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=10)
    plt.colorbar(scatter, ax=ax, label="Activation", shrink=0.7)

    return fig


def plot_feature_topomap(
    sae_feature_acts: np.ndarray,
    feature_idx: int,
    channel_names: List[str],
    n_time_patches: int,
    aggregation: str = "mean",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot topomap of a single SAE feature, aggregating over time patches.

    Args:
        sae_feature_acts: [n_tokens, d_sae]
        feature_idx: Which feature to visualize
        channel_names: Channel names
        n_time_patches: Number of time patches
        aggregation: How to aggregate over time: "mean" | "max" | "sum"
        title: Optional title
        save_path: Optional file path to save
    """
    n_channels = len(channel_names)
    acts = sae_feature_acts[:, feature_idx][:n_channels * n_time_patches]
    acts = acts.reshape(n_channels, n_time_patches)

    if aggregation == "mean":
        per_channel = acts.mean(axis=1)
    elif aggregation == "max":
        per_channel = acts.max(axis=1)
    elif aggregation == "sum":
        per_channel = acts.sum(axis=1)
    else:
        per_channel = acts.mean(axis=1)

    fig = plot_topomap(
        per_channel, channel_names,
        title=title or f"Feature {feature_idx} Topomap ({aggregation})",
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ── Per-class comparison ────────────────────────────────────────────────────


def plot_per_class_activation(
    per_class_mean_acts: np.ndarray,
    feature_idx: int,
    class_names: Optional[Dict[int, str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of per-class mean activation for a single SAE feature.

    Args:
        per_class_mean_acts: [n_classes, d_sae]
        feature_idx: Which feature to visualize
        class_names: Optional mapping from class index to name
    """
    if class_names is None:
        class_names = BCIC2A_CLASS_NAMES

    n_classes = per_class_mean_acts.shape[0]
    values = per_class_mean_acts[:, feature_idx]
    names = [class_names.get(i, f"Class {i}") for i in range(n_classes)]
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, values, color=colors[:n_classes], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Mean SAE Feature Activation", fontsize=11)
    ax.set_title(title or f"SAE Feature {feature_idx}: Per-Class Activation", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_per_class_activation_plotly(
    per_class_mean_acts: np.ndarray,
    feature_idx: int,
    class_names: Optional[Dict[int, str]] = None,
) -> go.Figure:
    """Interactive Plotly bar chart of per-class activation for a feature."""
    if class_names is None:
        class_names = BCIC2A_CLASS_NAMES

    n_classes = per_class_mean_acts.shape[0]
    values = per_class_mean_acts[:, feature_idx]
    names = [class_names.get(i, f"Class {i}") for i in range(n_classes)]
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker_color=colors[:n_classes],
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"SAE Feature {feature_idx}: Per-Class Activation",
        yaxis_title="Mean Activation",
        template="plotly_white",
    )
    return fig


# ── Top-K class-specific features ──────────────────────────────────────────


def get_top_class_features(
    per_class_mean_acts: np.ndarray,
    class_idx: int,
    top_k: int = 20,
    class_names: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """Get top-K features for a specific class.

    Returns DataFrame with columns: feature_idx, activation, selectivity, rank.
    """
    if class_names is None:
        class_names = BCIC2A_CLASS_NAMES

    acts = per_class_mean_acts[class_idx]
    n_classes = per_class_mean_acts.shape[0]
    sorted_idx = np.argsort(acts)[::-1][:top_k]

    rows = []
    for rank, idx in enumerate(sorted_idx):
        max_val = acts[idx]
        others_mean = (per_class_mean_acts[:, idx].sum() - max_val) / max(n_classes - 1, 1)
        selectivity = (max_val - others_mean) / (max_val + others_mean + 1e-9)
        rows.append({
            "rank": rank + 1,
            "feature_idx": int(idx),
            "activation": float(max_val),
            "selectivity": float(selectivity),
            "preferred_class": class_names.get(int(per_class_mean_acts[:, idx].argmax()), "Unknown"),
        })
    return pd.DataFrame(rows)


# ── EEG trial visualization ────────────────────────────────────────────────


def plot_eeg_trial(
    eeg: np.ndarray,
    channel_names: List[str],
    sample_rate: int = 250,
    title: str = "EEG Trial",
    highlight_channels: Optional[List[str]] = None,
    scale: float = 1.0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot raw EEG trial as multi-channel time series.

    Args:
        eeg: [C, T] EEG data
        channel_names: Channel names
        sample_rate: Sample rate in Hz
        title: Plot title
        highlight_channels: Optional list of channels to highlight in red
        scale: Vertical scale factor for traces
    """
    n_channels, n_samples = eeg.shape
    times = np.arange(n_samples) / sample_rate

    fig, ax = plt.subplots(figsize=(14, max(6, n_channels * 0.3)))

    offsets = np.arange(n_channels) * scale

    for i, ch_name in enumerate(channel_names):
        color = "red" if highlight_channels and ch_name in highlight_channels else "#333333"
        lw = 1.5 if highlight_channels and ch_name in highlight_channels else 0.8
        ax.plot(times, eeg[i] + offsets[i], color=color, linewidth=lw)

    ax.set_yticks(offsets)
    ax.set_yticklabels(channel_names, fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(times[0], times[-1])
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_eeg_trial_plotly(
    eeg: np.ndarray,
    channel_names: List[str],
    sample_rate: int = 250,
    title: str = "EEG Trial",
) -> go.Figure:
    """Interactive Plotly plot of raw EEG."""
    n_channels, n_samples = eeg.shape
    times = np.arange(n_samples) / sample_rate

    fig = go.Figure()
    for i, ch in enumerate(channel_names):
        offset = i * 1.0
        fig.add_trace(go.Scatter(
            x=times, y=eeg[i] + offset,
            mode="lines", name=ch,
            line=dict(width=1),
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis=dict(
            tickvals=[i * 1.0 for i in range(n_channels)],
            ticktext=channel_names,
            autorange="reversed",
        ),
        template="plotly_white",
        height=max(400, n_channels * 30),
        showlegend=False,
    )
    return fig


# ── Activation distribution ────────────────────────────────────────────────


def plot_activation_distribution(
    sae_feature_acts: np.ndarray,
    top_k: int = 10,
    title: str = "SAE Feature Activations",
) -> go.Figure:
    """Interactive bar chart of SAE feature activations for a single trial.

    Args:
        sae_feature_acts: [d_sae] mean-pooled activation for one trial
        top_k: Number of top features to annotate
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.arange(len(sae_feature_acts)),
        y=sae_feature_acts,
        mode="lines",
        name="Activation",
        line=dict(color="#3498db"),
    ))

    top_idx = np.argsort(sae_feature_acts)[::-1][:top_k]
    for idx in top_idx:
        fig.add_annotation(
            x=int(idx), y=float(sae_feature_acts[idx]),
            text=str(idx), showarrow=True, arrowhead=2,
            ax=0, ay=-20, arrowcolor="#e74c3c", opacity=0.8,
        )

    fig.update_layout(
        title=title,
        xaxis_title="SAE Latent Index",
        yaxis_title="Activation Value",
        template="plotly_white",
    )
    return fig


# ── Multi-feature comparison ───────────────────────────────────────────────


def plot_feature_comparison_grid(
    sae_feature_acts: np.ndarray,
    feature_indices: List[int],
    channel_names: List[str],
    n_time_patches: int,
    per_class_mean_acts: Optional[np.ndarray] = None,
    class_names: Optional[Dict[int, str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a grid comparing multiple SAE features side by side.

    Each row shows for one feature: channel×time heatmap, topomap, per-class bar.

    Args:
        sae_feature_acts: [n_tokens, d_sae]
        feature_indices: List of feature indices to compare
        channel_names: Channel names
        n_time_patches: Number of time patches
        per_class_mean_acts: [n_classes, d_sae]
        class_names: Optional mapping
    """
    if class_names is None:
        class_names = BCIC2A_CLASS_NAMES

    n_feats = len(feature_indices)
    n_channels = len(channel_names)
    has_class = per_class_mean_acts is not None
    n_cols = 3 if has_class else 2

    fig, axes = plt.subplots(n_feats, n_cols, figsize=(5 * n_cols, 3 * n_feats))
    if n_feats == 1:
        axes = axes[np.newaxis, :]

    for row, feat_idx in enumerate(feature_indices):
        acts = sae_feature_acts[:, feat_idx][:n_channels * n_time_patches]
        heatmap = acts.reshape(n_channels, n_time_patches)

        # Column 1: Channel × Time heatmap
        ax = axes[row, 0]
        im = ax.imshow(heatmap, aspect="auto", cmap="RdYlBu_r", interpolation="nearest")
        ax.set_yticks(range(n_channels))
        ax.set_yticklabels(channel_names, fontsize=7)
        ax.set_xlabel("Time patch")
        ax.set_title(f"Feature {feat_idx}", fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Column 2: Topomap
        ax2 = axes[row, 1]
        per_ch = heatmap.mean(axis=1)
        xs = [CHANNEL_POS_2D[ch][0] for ch in channel_names if ch in CHANNEL_POS_2D]
        ys = [CHANNEL_POS_2D[ch][1] for ch in channel_names if ch in CHANNEL_POS_2D]
        ch_vals = [per_ch[i] for i, ch in enumerate(channel_names) if ch in CHANNEL_POS_2D]
        head = plt.Circle((0, 0.2), 0.75, fill=False, linewidth=1.5, color="black")
        ax2.add_patch(head)
        ax2.plot([-0.06, 0, 0.06], [0.93, 1.0, 0.93], "k-", linewidth=1.5)
        vmin, vmax = min(ch_vals), max(ch_vals)
        if vmin == vmax:
            vmax = vmin + 1e-6
        sc = ax2.scatter(xs, ys, c=ch_vals, cmap="RdYlBu_r", s=200,
                         edgecolors="black", linewidths=1, vmin=vmin, vmax=vmax)
        ax2.set_xlim(-1.1, 1.1)
        ax2.set_ylim(-0.7, 1.15)
        ax2.set_aspect("equal")
        ax2.axis("off")
        ax2.set_title(f"Topomap", fontsize=11)
        plt.colorbar(sc, ax=ax2, shrink=0.7)

        # Column 3: Per-class bar
        if has_class:
            ax3 = axes[row, 2]
            n_classes = per_class_mean_acts.shape[0]
            vals = per_class_mean_acts[:, feat_idx]
            names = [class_names.get(i, f"C{i}") for i in range(n_classes)]
            colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
            ax3.bar(names, vals, color=colors[:n_classes], edgecolor="black", linewidth=0.5)
            ax3.set_ylabel("Mean act")
            ax3.set_title("Per-class", fontsize=11)
            ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig

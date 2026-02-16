"""
EEG-SAE Interactive Demo — Gradio Web Application

Explore how Sparse Autoencoder features activate on EEG data from the REVE
transformer. Upload or select EEG trials and interactively inspect which SAE
features fire, their spatial (channel) and temporal patterns, and per-class
specificity.

Usage:
    python -m src.demo.app \
        --sae_path checkpoints/sae_latest.pt \
        --feature_data_path results/feature_data/sae_feature_data.npz \
        --db_path /path/to/bcic2a.lmdb

Or:
    bash scripts/05_run_demo.sh
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.utils import (
    BCIC2A_CLASS_NAMES,
    CHANNEL_POS_2D,
    plot_activation_distribution,
    plot_channel_time_activation_plotly,
    plot_eeg_trial_plotly,
    plot_per_class_activation_plotly,
)
from src.demo.utils import load_dataset_for_demo, load_sae_tester

# ── Globals (set in main) ───────────────────────────────────────────────
sae_tester = None
eeg_dataset = None
CHANNEL_NAMES = []
N_TIME_PATCHES = 5


# ── Helper functions ────────────────────────────────────────────────────


def load_trial_by_index(trial_idx: int, split: str = "test"):
    """Load a specific trial from the dataset."""
    loader = eeg_dataset[split]
    dataset = loader.dataset
    if trial_idx >= len(dataset):
        return None, None
    eeg, label = dataset[trial_idx]
    return eeg.numpy(), int(label)


def get_trial_list(split: str = "test", max_trials: int = 200):
    """Get list of trial descriptions for dropdown."""
    loader = eeg_dataset[split]
    dataset = loader.dataset
    n = min(len(dataset), max_trials)
    items = []
    for i in range(n):
        _, label = dataset[i]
        cls_name = BCIC2A_CLASS_NAMES.get(int(label), f"Class {label}")
        items.append(f"Trial {i} — {cls_name}")
    return items


# ── Callback functions ──────────────────────────────────────────────────


def on_trial_selected(trial_selection: str, split: str):
    """When a trial is selected, run SAE and return all plots."""
    if not trial_selection:
        return None, None, None, None, None, None, []

    # Parse trial index
    trial_idx = int(trial_selection.split(" ")[1])
    eeg, label = load_trial_by_index(trial_idx, split)
    if eeg is None:
        return None, None, None, None, None, None, []

    cls_name = BCIC2A_CLASS_NAMES.get(label, f"Class {label}")

    # Register trial in SAE tester
    sae_tester.register_trial(eeg, label)

    # 1. Raw EEG plot
    eeg_plot = plot_eeg_trial_plotly(
        eeg, CHANNEL_NAMES, sample_rate=250,
        title=f"Trial {trial_idx} — {cls_name}",
    )

    # 2. Activation distribution
    mean_act = sae_tester.get_activation_distribution()
    act_plot = plot_activation_distribution(
        mean_act, top_k=10,
        title=f"SAE Activation Distribution — {cls_name}",
    )

    # 3. Get top features
    top_feats, top_vals = sae_tester.get_top_features_trial(top_k=10)

    # 4. Reconstruction metrics
    recon = sae_tester.get_reconstruction_error()
    recon_md = (
        f"**Reconstruction Quality:**\n"
        f"- MSE: {recon['mse']:.6f}\n"
        f"- Relative MSE: {recon['relative_mse']:.4f}\n"
        f"- Cosine Similarity: {recon['cosine_similarity']:.4f}\n"
        f"- L0 (avg active features/token): {recon['L0']:.1f}"
    )

    # 5. Build radio choices for top features
    radio_choices = []
    for feat_idx, feat_val in zip(top_feats, top_vals):
        info = sae_tester.get_feature_info(feat_idx)
        pref = info["preferred_class_name"]
        radio_choices.append(
            f"Feature {feat_idx} — act={feat_val:.4f} sel={info['selectivity']:.3f} ({pref})"
        )

    # 6. Default: show first feature details
    if len(top_feats) > 0:
        heatmap, topomap, class_bar, info_md = _generate_feature_plots(int(top_feats[0]))
    else:
        heatmap, topomap, class_bar, info_md = None, None, None, ""

    # Return: eeg_plot, act_plot, recon_md, heatmap, class_bar, info_md, radio_choices
    return eeg_plot, act_plot, recon_md, heatmap, class_bar, info_md, gr.update(choices=radio_choices, value=radio_choices[0] if radio_choices else None)


def on_feature_selected(feature_choice: str):
    """When a feature is selected from the radio list, update its plots."""
    if not feature_choice:
        return None, None, ""

    feat_idx = int(feature_choice.split(" ")[1])
    heatmap, topomap, class_bar, info_md = _generate_feature_plots(feat_idx)
    return heatmap, class_bar, info_md


def on_channel_clicked(channel_name: str, top_k: int = 5):
    """When a channel is clicked, show its top features."""
    if not channel_name or sae_tester._eeg is None:
        return ""

    try:
        top_feats, top_vals = sae_tester.get_top_features_channel(channel_name, top_k)
    except ValueError:
        return f"Channel '{channel_name}' not found."

    lines = [f"### Top {top_k} features for channel **{channel_name}**\n"]
    for feat_idx, feat_val in zip(top_feats, top_vals):
        info = sae_tester.get_feature_info(feat_idx)
        lines.append(
            f"- **Feature {feat_idx}**: act={feat_val:.4f}, "
            f"selectivity={info['selectivity']:.3f}, "
            f"preferred={info['preferred_class_name']}"
        )
    return "\n".join(lines)


def _generate_feature_plots(feature_idx: int):
    """Generate channel×time heatmap, per-class bar, and info markdown for a feature."""
    token_acts = sae_tester.get_token_activations()

    # Heatmap
    heatmap = plot_channel_time_activation_plotly(
        token_acts, feature_idx, CHANNEL_NAMES, N_TIME_PATCHES,
        title=f"Feature {feature_idx}: Channel × Time Activation",
    )

    # Topomap (as plotly scatter)
    channel_acts = sae_tester.get_channel_activation(feature_idx, aggregation="mean")
    topomap = _plotly_topomap(channel_acts, CHANNEL_NAMES, f"Feature {feature_idx} Topomap")

    # Per-class bar
    class_bar = plot_per_class_activation_plotly(
        sae_tester.stats["per_class_mean_acts"].numpy(), feature_idx,
    )

    # Info markdown
    info = sae_tester.get_feature_info(feature_idx)
    info_md = (
        f"### Feature {feature_idx}\n"
        f"- **Mean Activation**: {info['mean_act']:.6f}\n"
        f"- **Sparsity**: {info['sparsity']:.6f}\n"
        f"- **Class Entropy**: {info['entropy']:.4f}\n"
        f"- **Selectivity**: {info['selectivity']:.4f}\n"
        f"- **Preferred Class**: {info['preferred_class_name']}\n"
        f"- **Per-class acts**: {', '.join(f'{v:.4f}' for v in info['per_class_acts'])}"
    )

    return heatmap, topomap, class_bar, info_md


def _plotly_topomap(
    channel_values: np.ndarray,
    channel_names: List[str],
    title: str,
) -> go.Figure:
    """Create an interactive Plotly scatter topomap."""
    xs, ys, vals, names = [], [], [], []
    for ch, val in zip(channel_names, channel_values):
        if ch in CHANNEL_POS_2D:
            x, y = CHANNEL_POS_2D[ch]
            xs.append(x)
            ys.append(y)
            vals.append(float(val))
            names.append(ch)

    fig = go.Figure()

    # Head outline
    theta = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(go.Scatter(
        x=0.75 * np.cos(theta), y=0.2 + 0.75 * np.sin(theta),
        mode="lines", line=dict(color="black", width=2),
        showlegend=False, hoverinfo="skip",
    ))
    # Nose
    fig.add_trace(go.Scatter(
        x=[-0.06, 0, 0.06], y=[0.93, 1.0, 0.93],
        mode="lines", line=dict(color="black", width=2),
        showlegend=False, hoverinfo="skip",
    ))

    # Channel markers
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        marker=dict(
            size=30, color=vals, colorscale="RdYlBu_r",
            colorbar=dict(title="Activation"),
            line=dict(color="black", width=1),
        ),
        text=names, textposition="middle center",
        textfont=dict(size=8),
        hovertext=[f"{n}: {v:.4f}" for n, v in zip(names, vals)],
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(range=[-1.1, 1.1], visible=False),
        yaxis=dict(range=[-0.7, 1.15], scaleanchor="x", visible=False),
        template="plotly_white",
        width=400, height=450,
    )
    return fig


def get_feature_overview_data():
    """Return feature overview data for the stats tab."""
    if sae_tester is None:
        return None, ""

    from analysis.utils import dead_feature_summary, feature_overview_scatter

    summary = dead_feature_summary(sae_tester.stats)
    fig = feature_overview_scatter(sae_tester.stats, mask=summary["alive_mask"])

    md = (
        f"### Feature Summary\n"
        f"- Total features: {summary['n_total']}\n"
        f"- Alive: {summary['n_alive']} ({1 - summary['frac_dead']:.1%})\n"
        f"- Dead: {summary['n_dead']} ({summary['frac_dead']:.1%})"
    )
    return fig, md


def get_class_top_features(class_idx: int, top_k: int = 15):
    """Get top features table for a specific class."""
    if sae_tester is None:
        return ""

    from analysis.utils import get_top_class_features

    df = get_top_class_features(
        sae_tester.stats["per_class_mean_acts"].numpy(),
        class_idx, top_k=top_k,
    )
    cls_name = BCIC2A_CLASS_NAMES.get(class_idx, f"Class {class_idx}")
    header = f"### Top {top_k} features for **{cls_name}**\n\n"
    return header + df.to_markdown(index=False)


# ── Gradio App ──────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    """Build the Gradio Blocks app."""
    with gr.Blocks(
        title="EEG-SAE Explorer",
        theme=gr.themes.Soft(),
        css="""
        .top-panel { margin-bottom: 10px; }
        """,
    ) as app:
        gr.Markdown(
            "# EEG-SAE Explorer\n"
            "Interactive visualization of Sparse Autoencoder features on REVE EEG transformer activations.\n"
            "Select an EEG trial to see which SAE features fire and their spatial/temporal patterns."
        )

        with gr.Tabs():
            # ── Tab 1: Trial Explorer ────────────────────────────────
            with gr.Tab("Trial Explorer"):
                with gr.Row():
                    with gr.Column(scale=1):
                        split_dropdown = gr.Dropdown(
                            choices=["test", "train"],
                            value="test",
                            label="Dataset split",
                        )
                        trial_dropdown = gr.Dropdown(
                            choices=[], label="Select EEG Trial",
                            interactive=True,
                        )
                        load_btn = gr.Button("Load Trial", variant="primary")
                        recon_display = gr.Markdown("")

                    with gr.Column(scale=3):
                        eeg_plot = gr.Plot(label="Raw EEG Signal")

                with gr.Row():
                    act_plot = gr.Plot(label="SAE Activation Distribution")

                gr.Markdown("---")
                gr.Markdown("## Feature Inspector")
                gr.Markdown("Select a top-firing feature to see its spatial & temporal activation patterns.")

                with gr.Row():
                    with gr.Column(scale=1):
                        feature_radio = gr.Radio(
                            choices=[], label="Top SAE Features",
                        )
                        feature_info = gr.Markdown("")

                    with gr.Column(scale=2):
                        with gr.Row():
                            heatmap_plot = gr.Plot(label="Channel × Time Heatmap")
                            class_bar_plot = gr.Plot(label="Per-Class Activation")

                gr.Markdown("---")
                gr.Markdown("### Channel Inspector")
                with gr.Row():
                    channel_dropdown = gr.Dropdown(
                        choices=CHANNEL_NAMES if CHANNEL_NAMES else [],
                        label="Select Channel",
                    )
                    channel_info = gr.Markdown("")

                # ── Events ──
                def update_trial_list(split):
                    trials = get_trial_list(split)
                    return gr.update(choices=trials, value=trials[0] if trials else None)

                split_dropdown.change(
                    fn=update_trial_list,
                    inputs=[split_dropdown],
                    outputs=[trial_dropdown],
                )

                load_btn.click(
                    fn=on_trial_selected,
                    inputs=[trial_dropdown, split_dropdown],
                    outputs=[eeg_plot, act_plot, recon_display, heatmap_plot,
                             class_bar_plot, feature_info, feature_radio],
                )

                feature_radio.change(
                    fn=on_feature_selected,
                    inputs=[feature_radio],
                    outputs=[heatmap_plot, class_bar_plot, feature_info],
                )

                channel_dropdown.change(
                    fn=on_channel_clicked,
                    inputs=[channel_dropdown],
                    outputs=[channel_info],
                )

            # ── Tab 2: Feature Statistics ────────────────────────────
            with gr.Tab("Feature Statistics"):
                gr.Markdown("## SAE Feature Overview")
                gr.Markdown("Scatter plot of all alive features: sparsity vs mean activation, colored by class entropy.")

                overview_btn = gr.Button("Load Feature Overview", variant="primary")
                overview_md = gr.Markdown("")
                overview_plot = gr.Plot(label="Feature Scatter")

                overview_btn.click(
                    fn=get_feature_overview_data,
                    outputs=[overview_plot, overview_md],
                )

            # ── Tab 3: Per-Class Features ────────────────────────────
            with gr.Tab("Per-Class Features"):
                gr.Markdown("## Top Features by Motor Imagery Class")
                gr.Markdown("Discover which SAE features are most specific to each movement class.")

                with gr.Row():
                    class_dropdown = gr.Dropdown(
                        choices=[f"{BCIC2A_CLASS_NAMES[i]} (class {i})" for i in range(4)],
                        value=f"{BCIC2A_CLASS_NAMES[0]} (class 0)",
                        label="Select Class",
                    )
                    topk_slider = gr.Slider(5, 50, value=15, step=5, label="Top K")

                class_features_md = gr.Markdown("")

                def on_class_selected(cls_str, top_k):
                    cls_idx = int(cls_str.split("class ")[1].rstrip(")"))
                    return get_class_top_features(cls_idx, int(top_k))

                class_dropdown.change(
                    fn=on_class_selected,
                    inputs=[class_dropdown, topk_slider],
                    outputs=[class_features_md],
                )
                topk_slider.change(
                    fn=on_class_selected,
                    inputs=[class_dropdown, topk_slider],
                    outputs=[class_features_md],
                )

            # ── Tab 4: About ─────────────────────────────────────────
            with gr.Tab("About"):
                gr.Markdown("""
## About EEG-SAE Explorer

This interactive tool visualizes **Sparse Autoencoder (SAE)** features learned from the
internal activations of **REVE**, a pretrained EEG transformer.

### How it works

1. **REVE** processes raw EEG `[C, T]` into token representations `[C×H, d_model]`
   where each token corresponds to a `(channel, time-patch)` pair.
2. The **SAE** encodes these token representations into sparse features `[C×H, d_sae]`.
3. We analyze which features fire for each trial, their spatial patterns (which channels),
   temporal patterns (which time patches), and class specificity.

### Tabs

- **Trial Explorer**: Select an EEG trial to see raw signal, SAE activations, and
  per-feature channel×time heatmaps.
- **Feature Statistics**: Overview scatter plot of all features (sparsity vs activation).
- **Per-Class Features**: Discover class-specific features for motor imagery decoding.

### Key concepts

| Metric | Description |
|--------|-------------|
| **Sparsity** | Fraction of tokens where the feature fires |
| **Mean Activation** | Average activation strength across all tokens |
| **Class Entropy** | Low = class-specific, high = general |
| **Selectivity** | (max_class − mean_others) / (max_class + mean_others) |
| **L0** | Average number of active SAE features per token |

### References

- [PatchSAE](https://github.com/dynamical-inference/patchsae) — SAE for Vision Transformers
- [REVE](https://huggingface.co/brain-bzh/reve-base) — EEG Foundation Model
                """)

    return app


# ── Main entry point ────────────────────────────────────────────────────

def main():
    global sae_tester, eeg_dataset, CHANNEL_NAMES, N_TIME_PATCHES

    parser = argparse.ArgumentParser(description="EEG-SAE Interactive Demo")
    parser.add_argument("--sae_path", type=str, required=True,
                        help="Path to trained SAE checkpoint (.pt)")
    parser.add_argument("--feature_data_path", type=str, required=True,
                        help="Path to precomputed feature data (.npz)")
    parser.add_argument("--db_path", type=str, required=True,
                        help="Path to BCIC2a LMDB database")
    parser.add_argument("--model_name", type=str, default="brain-bzh/reve-base")
    parser.add_argument("--pos_bank_name", type=str, default="brain-bzh/reve-positions")
    parser.add_argument("--local_model_path", type=str, default=None)
    parser.add_argument("--block_layer", type=int, default=-2)
    parser.add_argument("--module_name", type=str, default="resid")
    parser.add_argument("--n_time_patches", type=int, default=5)
    parser.add_argument("--test_subject", type=int, default=9)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", default=False)
    args = parser.parse_args()

    N_TIME_PATCHES = args.n_time_patches

    # Load SAE tester
    sae_tester = load_sae_tester(
        sae_path=args.sae_path,
        feature_data_path=args.feature_data_path,
        model_name=args.model_name,
        pos_bank_name=args.pos_bank_name,
        local_model_path=args.local_model_path,
        block_layer=args.block_layer,
        module_name=args.module_name,
        n_time_patches=args.n_time_patches,
        device=args.device,
    )

    CHANNEL_NAMES = sae_tester.channel_names

    # Load dataset
    print(f"Loading BCIC2a dataset from {args.db_path}...")
    eeg_dataset = load_dataset_for_demo(
        db_path=args.db_path,
        batch_size=32,
        test_subject_ids=[args.test_subject],
    )

    # Build and launch
    app = build_app()
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

# EEG-SAE: Sparse Autoencoders for EEG Transformers

Mechanistic interpretability for EEG foundation models using Sparse Autoencoders (SAEs). This project applies the [PatchSAE](https://github.com/dynamical-inference/patchsae) methodology — originally designed for Vision Transformers — to EEG transformers, specifically the [REVE](https://huggingface.co/brain-bzh/reve-base) model.

## Overview

**What this does:** Trains sparse autoencoders on the internal activations of REVE (a pretrained EEG transformer) to discover interpretable features that the model uses to represent EEG signals. These features can then be analyzed, probed, and used for downstream classification tasks like motor imagery decoding (BCIC2a).

**Key idea:** Just as PatchSAE decomposes CLIP ViT patch representations into sparse, interpretable features for images, EEG-SAE decomposes REVE token representations into sparse features for EEG signals — each token corresponds to a (channel, time-patch) pair.

### Architecture

```
Raw EEG [B, C, T]
    │
    ▼
┌──────────────────────┐
│  REVE Encoder        │  Pretrained EEG transformer
│  (Hooked)            │  with hook points at every layer
│                      │
│  Patch embedding     │  Unfold: T → H patches per channel
│  + Fourier 4D PE     │  Position: (x, y, z, t) encoding
│  + Transformer       │  Depth: 22 blocks (base)
│                      │
│  Hook: resid_post    │◄── Extract activations here
└──────────────────────┘
    │
    ▼  Activations [B, C*H, d_model]
    │
┌──────────────────────┐
│  Sparse Autoencoder  │  Overcomplete dictionary
│                      │
│  Encode: d_in → d_sae│  d_sae = expansion_factor × d_in
│  ReLU activation     │  Sparse feature activations
│  Decode: d_sae → d_in│  Reconstruct with unit-norm decoder
│                      │
│  Loss: MSE + L1      │  + ghost gradients for dead neurons
└──────────────────────┘
    │
    ▼  Sparse features [B, C*H, d_sae]
    │
┌──────────────────────┐
│  Downstream Tasks    │
│                      │
│  • Linear probe      │  Logistic regression on SAE features
│  • Top-K masking     │  Keep only top-K class-specific features
│  • Feature analysis  │  Sparsity, per-class activation stats
└──────────────────────┘
```

## Project Structure

```
mech-interp-eeg/
├── models/reve/                    # Hooked REVE model (already implemented)
│   ├── configuration_reve.py       # REVE config
│   ├── modeling_reve.py            # REVE with HookPoints
│   ├── hooked_reve.py              # SAE attachment (HookedSAEReve)
│   └── __init__.py
│
├── src/sae_training/               # SAE training infrastructure
│   ├── config.py                   # EEGSAERunnerConfig dataclass
│   ├── sparse_autoencoder.py       # SAE model (encode/decode/loss)
│   ├── hooked_eeg_transformer.py   # Wrapper for activation extraction
│   ├── eeg_activations_store.py    # Streams EEG → activations
│   ├── sae_trainer.py              # Training loop with ghost grads
│   └── utils.py                    # LR schedulers
│
├── src/demo/                       # Interactive web demo
│   ├── app.py                      # Gradio web app (main entry)
│   ├── core.py                     # SAETester analysis engine
│   └── utils.py                    # Loading helpers
│
├── analysis/                       # Analysis & visualization
│   ├── utils.py                    # Plotting, topomaps, entropy, etc.
│   └── analysis.ipynb              # Interactive analysis notebook
│
├── tasks/                          # Entry-point scripts
│   ├── train_sae_eeg.py            # Train SAE on REVE activations
│   ├── compute_sae_feature_data.py # Compute feature statistics
│   └── utils.py                    # Channel configs
│
├── downstream/                     # Downstream evaluation
│   ├── bcic2a_dataset.py           # BCIC2a LMDB dataset loader
│   └── linear_probe.py             # Linear probe + top-K masking
│
├── scripts/                        # Shell scripts
│   ├── 01_run_train.sh             # Train SAE
│   ├── 02_run_compute_features.sh  # Compute feature data
│   ├── 03_run_bcic2a_probe.sh      # Evaluate single subject
│   ├── 04_run_all_subjects.sh      # Evaluate all 9 subjects
│   └── 05_run_demo.sh              # Launch interactive web demo
│
├── REVE_Tutorial_EEGMAT.ipynb      # REVE usage tutorial
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/mech-interp-eeg.git
cd mech-interp-eeg

# Install dependencies
pip install -r requirements.txt

# Authenticate with HuggingFace (REVE is a gated model)
huggingface-cli login
```

### Dependencies

- **PyTorch** ≥ 2.2.0
- **transformers** — loads REVE from HuggingFace
- **transformer_lens** — `HookPoint` / `HookedRootModule`
- **sae_lens** — SAE configuration types
- **einops** — tensor reshaping
- **geom_median** — geometric median for bias init
- **lmdb** — BCIC2a dataset storage
- **scikit-learn** — linear probe (logistic regression)
- **wandb** — optional logging
- **gradio** ≥ 4.0 — interactive web demo
- **plotly** — interactive plots (analysis + demo)
- **matplotlib** — static plots (analysis notebook)
- **pandas** — feature tables

## Quick Start

### 1. Prepare Data

The BCIC2a dataset should be preprocessed into an LMDB database following the [STELAR](https://github.com/your-org/STELAR-private) preprocessing pipeline. Each sample is stored as:

```python
{"sample": np.ndarray[C, T], "label": int}  # C=22 channels, T=1024 samples
```

Set the path:
```bash
export BCIC2A_DB_PATH=/path/to/bcic2a.lmdb
```

### 2. Train SAE

```bash
# Train SAE on REVE activations (second-to-last layer)
bash scripts/01_run_train.sh

# Or run directly with custom args:
python tasks/train_sae_eeg.py \
    --db_path $BCIC2A_DB_PATH \
    --model_name brain-bzh/reve-base \
    --block_layer -2 \
    --expansion_factor 16 \
    --l1_coefficient 1e-4 \
    --batch_size 32 \
    --total_training_tokens 500000 \
    --device cuda
```

**Key hyperparameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_layer` | -2 | Transformer block to extract from (negative = from end) |
| `expansion_factor` | 16 | Dictionary size multiplier (d_sae = 16 × 512 = 8192) |
| `l1_coefficient` | 1e-4 | Sparsity penalty weight |
| `batch_size` | 32 | Training batch size |
| `total_training_tokens` | 500,000 | Total training samples |
| `use_ghost_grads` | True | Ghost gradients for dead neuron revival |
| `lr` | 3e-4 | Learning rate |

### 3. Compute Feature Statistics

```bash
bash scripts/02_run_compute_features.sh
```

This outputs an `.npz` file with:
- `feature_mean_acts` — mean activation per SAE feature
- `feature_sparsity` — fraction of tokens where each feature fires
- `per_class_mean_acts` — per-class mean activation for motor imagery classes
- `top_activating_trial_indices` — which EEG trials most activate each feature

### 4. Evaluate with Linear Probe (BCIC2a)

```bash
# Single subject
TEST_SUBJECT=9 bash scripts/03_run_bcic2a_probe.sh

# All subjects (leave-one-out)
bash scripts/04_run_all_subjects.sh
```

This evaluates:
1. **SAE feature probe** — logistic regression on SAE features (mean-pooled)
2. **Raw feature probe** — same but on raw REVE activations (baseline)
3. **Top-K masking** — classification using only the K most class-specific SAE features

Results are saved as JSON in `results/bcic2a/`.

### 5. Interactive Analysis

#### Analysis Notebook

Open [analysis/analysis.ipynb](analysis/analysis.ipynb) for interactive exploration:
- Feature overview scatter plot (sparsity vs activation, colored by class entropy)
- Dead/alive feature summary
- Class selectivity distribution
- Per-feature channel × time heatmaps
- Scalp topomaps showing spatial patterns
- Per-class activation bar charts
- Multi-feature comparison grids

#### Web Demo (Gradio)

Launch the interactive web app:

```bash
bash scripts/05_run_demo.sh

# Or with custom args:
python -m src.demo.app \
    --sae_path checkpoints/sae_latest.pt \
    --feature_data_path results/feature_data/sae_feature_data.npz \
    --db_path $BCIC2A_DB_PATH \
    --device cpu \
    --port 7860
```

The web demo provides three tabs:

| Tab | Description |
|-----|-------------|
| **Trial Explorer** | Select EEG trial → see raw signal, SAE activation distribution, per-feature channel×time heatmaps, topomaps, and per-class bars |
| **Feature Statistics** | Overview scatter plot of all alive features |
| **Per-Class Features** | Top-K class-specific feature tables |

Features of the Trial Explorer:
- **Raw EEG plot**: Interactive multi-channel time series
- **Activation distribution**: Which SAE features fire (top-10 annotated)
- **Reconstruction metrics**: MSE, cosine similarity, L0 sparsity
- **Channel × Time heatmap**: Spatial-temporal activation pattern for each feature
- **Per-class bar chart**: How a feature splits across motor imagery classes
- **Channel inspector**: Select a channel to see its top features

## How It Works

### SAE Training

The SAE learns a sparse overcomplete dictionary from REVE's residual stream:

1. **Extract activations**: Run EEG through REVE, hook the residual stream at block `L` → `[B, N, d_in]` where `N = n_channels × n_time_patches`
2. **Encode**: `z = ReLU(W_enc @ (x - b_dec) + b_enc)` → sparse codes `[B, N, d_sae]`
3. **Decode**: `x̂ = W_dec @ z + b_dec` → reconstructed activations
4. **Loss**: $\mathcal{L} = \text{MSE}(\hat{x}, x) + \lambda \cdot \|z\|_1$
5. **Ghost gradients**: Revive dead neurons by injecting gradient signal through exponential pre-activations

### Downstream Evaluation

For BCIC2a 4-class motor imagery classification:

1. Encode all training/test EEG trials with the trained SAE
2. Pool SAE features over tokens (mean/max) → feature vector per trial
3. Train logistic regression on training features
4. Evaluate accuracy, Cohen's kappa, F1 on test subject

### Top-K Feature Masking

Inspired by PatchSAE's interpretability analysis:

1. Compute mean SAE feature activation per motor imagery class
2. Select top-K features for each class (union of all classes)
3. Zero out all other SAE features
4. Re-evaluate classification — measures whether a small number of features capture class-discriminative information

## Available Hook Points

The hooked REVE model provides hook points at every layer:

```python
from src.sae_training.hooked_eeg_transformer import HookedEEGTransformer

model = HookedEEGTransformer(model_name="brain-bzh/reve-base", device="cuda")

# List all hooks
print(model.get_all_hook_names())

# Hooks per layer:
# transformer.layers.{i}.hook_resid_pre    — before attention
# transformer.layers.{i}.hook_resid_mid    — after attention, before MLP
# transformer.layers.{i}.hook_resid_post   — after MLP (residual stream)
# transformer.layers.{i}.attn.hook_q       — attention queries
# transformer.layers.{i}.attn.hook_k       — attention keys
# transformer.layers.{i}.attn.hook_v       — attention values
# transformer.layers.{i}.attn.hook_attn_out— attention output
# transformer.layers.{i}.ff.hook_mlp_in    — MLP input
# transformer.layers.{i}.ff.hook_mlp_out   — MLP output
```

## Differences from PatchSAE

| Aspect | PatchSAE (Vision) | EEG-SAE (This project) |
|--------|-------------------|----------------------|
| Model | CLIP ViT | REVE EEG Transformer |
| Input | Images [B, 3, H, W] | EEG [B, C, T] |
| Tokens | Image patches | (channel, time-patch) pairs |
| Positional encoding | Learned 2D | Fourier 4D (x, y, z, t) |
| Dataset | ImageNet | BCIC2a (motor imagery) |
| Downstream | Zero-shot classification | Linear probe |
| Class token | Yes (CLS) | No (all patch tokens) |

## Citation

If you use this work, please cite:

```bibtex
@article{patchsae,
    title={Sparse Autoencoders for Vision Transformers},
    author={...},
    year={2024},
}

@article{reve,
    title={REVE: A Foundation Model for EEG},
    author={...},
    year={2024},
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
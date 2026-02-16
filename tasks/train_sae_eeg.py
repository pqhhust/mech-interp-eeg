"""
Train a Sparse Autoencoder on REVE EEG transformer activations.

Usage:
    python tasks/train_sae_eeg.py \
        --db_path /path/to/bcic2a.lmdb \
        --model_name brain-bzh/reve-base \
        --block_layer -2 \
        --expansion_factor 16 \
        --batch_size 32 \
        --total_training_tokens 500000 \
        --device cuda
"""

import argparse
import os
import sys

import torch

from src.sae_training.config import EEGSAERunnerConfig
from src.sae_training.eeg_activations_store import EEGActivationsStore
from src.sae_training.hooked_eeg_transformer import HookedEEGTransformer
from src.sae_training.sae_trainer import SAETrainer
from src.sae_training.sparse_autoencoder import SparseAutoencoder
from src.sae_training.utils import get_scheduler
from tasks.utils import BCIC2A_CHANNELS, parse_channel_names


def main():
    parser = argparse.ArgumentParser(description="Train SAE on REVE EEG activations")

    # ── Data ──
    parser.add_argument("--db_path", type=str, required=True, help="LMDB dataset path")
    parser.add_argument("--scale_div", type=float, default=100.0)

    # ── Model ──
    parser.add_argument("--model_name", type=str, default="brain-bzh/reve-base")
    parser.add_argument("--pos_bank_name", type=str, default="brain-bzh/reve-positions")
    parser.add_argument("--local_model_path", type=str, default=None)
    parser.add_argument("--block_layer", type=int, default=-2)
    parser.add_argument("--module_name", type=str, default="resid")
    parser.add_argument("--d_in", type=int, default=512, help="REVE hidden dim (512 for base, 1024 for large)")

    # ── EEG Geometry ──
    parser.add_argument("--channel_names", type=str, default=None,
                        help="Comma-separated channel names (defaults to BCIC2a 22-ch)")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--patch_size", type=int, default=200)
    parser.add_argument("--patch_overlap", type=int, default=20)

    # ── SAE ──
    parser.add_argument("--expansion_factor", type=int, default=16)
    parser.add_argument("--b_dec_init_method", type=str, default="geometric_median")
    parser.add_argument("--class_token", action="store_true", default=False)

    # ── Training ──
    parser.add_argument("--l1_coefficient", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_scheduler_name", type=str, default="constantwithwarmup")
    parser.add_argument("--lr_warm_up_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--total_training_tokens", type=int, default=500_000)

    # ── Ghost grads / Dead neurons ──
    parser.add_argument("--use_ghost_grads", action="store_true", default=True)
    parser.add_argument("--no_ghost_grads", action="store_true", default=False)
    parser.add_argument("--feature_sampling_window", type=int, default=500)
    parser.add_argument("--dead_feature_window", type=int, default=250)
    parser.add_argument("--dead_feature_threshold", type=float, default=1e-8)

    # ── Logging ──
    parser.add_argument("--log_to_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="eeg-sae")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_log_frequency", type=int, default=10)

    # ── Misc ──
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_checkpoints", type=int, default=2)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints")

    args = parser.parse_args()

    # Handle ghost grads flag
    use_ghost_grads = args.use_ghost_grads and not args.no_ghost_grads

    # Parse channels
    channels = parse_channel_names(args.channel_names) or BCIC2A_CHANNELS
    n_channels = len(channels)

    # ── Build config ──
    cfg = EEGSAERunnerConfig(
        model_name=args.model_name,
        pos_bank_name=args.pos_bank_name,
        local_model_path=args.local_model_path,
        module_name=args.module_name,
        block_layer=args.block_layer,
        channel_names=channels,
        n_channels=n_channels,
        seq_len=args.seq_len,
        patch_size=args.patch_size,
        patch_overlap=args.patch_overlap,
        d_in=args.d_in,
        expansion_factor=args.expansion_factor,
        b_dec_init_method=args.b_dec_init_method,
        class_token=args.class_token,
        dataset_path=args.db_path,
        total_training_tokens=args.total_training_tokens,
        scale_div=args.scale_div,
        l1_coefficient=args.l1_coefficient,
        lr=args.lr,
        lr_scheduler_name=args.lr_scheduler_name,
        lr_warm_up_steps=args.lr_warm_up_steps,
        batch_size=args.batch_size,
        use_ghost_grads=use_ghost_grads,
        feature_sampling_window=args.feature_sampling_window,
        dead_feature_window=args.dead_feature_window,
        dead_feature_threshold=args.dead_feature_threshold,
        log_to_wandb=args.log_to_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_log_frequency=args.wandb_log_frequency,
        device=args.device,
        seed=args.seed,
        n_checkpoints=args.n_checkpoints,
        checkpoint_path=args.checkpoint_path,
        dtype=torch.float32,
    )

    # ── Load REVE model ──
    print("Loading REVE model...")
    model = HookedEEGTransformer(
        model_name=args.model_name,
        pos_bank_name=args.pos_bank_name,
        channel_names=channels,
        local_model_path=args.local_model_path,
        device=args.device,
    )
    print(f"REVE loaded: embed_dim={model.embed_dim}, depth={model.depth}")

    # ── Create SAE ──
    print("Creating SAE...")
    sae = SparseAutoencoder(cfg, args.device)
    print(f"SAE: d_in={sae.d_in}, d_sae={sae.d_sae}")

    # ── Create activation store ──
    print("Creating activation store...")
    activation_store = EEGActivationsStore(cfg, model, db_path=args.db_path)

    # ── Initialize b_dec ──
    print("Initializing SAE b_dec...")
    sae.initialize_b_dec(activation_store)
    sae.train()

    # ── Optimizer + scheduler ──
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    total_steps = cfg.total_training_tokens // cfg.batch_size
    scheduler = get_scheduler(
        args.lr_scheduler_name,
        optimizer=optimizer,
        warm_up_steps=args.lr_warm_up_steps,
        training_steps=total_steps,
    )

    # ── W&B ──
    if cfg.log_to_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, config=cfg.__dict__, name=cfg.run_name)

    # ── Train ──
    trainer = SAETrainer(sae, model, activation_store, cfg, optimizer, scheduler, args.device)
    trained_sae = trainer.fit()

    print(f"\nTraining complete! Final checkpoint saved to {cfg.checkpoint_path}/")


if __name__ == "__main__":
    main()

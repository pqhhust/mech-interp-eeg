"""
SAE Trainer for EEG Transformer activations.

Adapted from PatchSAE's SAETrainer for EEG/REVE. Runs the training loop,
tracks feature sparsity, handles ghost gradients and dead neuron resampling,
and optionally logs to Weights & Biases.
"""

from typing import Any, Dict

import torch
import wandb
from tqdm import tqdm

from src.sae_training.config import EEGSAERunnerConfig
from src.sae_training.eeg_activations_store import EEGActivationsStore
from src.sae_training.hooked_eeg_transformer import HookedEEGTransformer
from src.sae_training.sparse_autoencoder import SparseAutoencoder


class SAETrainer:
    """Trains a Sparse Autoencoder on EEG transformer activations."""

    def __init__(
        self,
        sae: SparseAutoencoder,
        model: HookedEEGTransformer,
        activation_store: EEGActivationsStore,
        cfg: EEGSAERunnerConfig,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
    ):
        self.sae = sae
        self.model = model
        self.activation_store = activation_store
        self.cfg = cfg
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.act_freq_scores = torch.zeros(cfg.d_sae, device=device)
        self.n_forward_passes_since_fired = torch.zeros(cfg.d_sae, device=device)
        self.n_frac_active_tokens = 0
        self.n_training_tokens = 0
        self.ghost_grad_neuron_mask = None
        self.n_training_steps = 0

        self.checkpoint_thresholds = list(
            range(
                0,
                cfg.total_training_tokens,
                max(1, cfg.total_training_tokens // max(1, cfg.n_checkpoints)),
            )
        )[1:]

    def _build_sparsity_log_dict(self) -> Dict[str, Any]:
        feature_freq = self.act_freq_scores / max(self.n_frac_active_tokens, 1)
        log_feature_freq = torch.log10(feature_freq + 1e-10).detach().cpu()
        return {
            "plots/feature_density_line_chart": wandb.Histogram(log_feature_freq.numpy()),
            "metrics/mean_log10_feature_sparsity": log_feature_freq.mean().item(),
        }

    @torch.no_grad()
    def _reset_running_sparsity_stats(self):
        self.act_freq_scores = torch.zeros(self.cfg.d_sae, device=self.device)
        self.n_frac_active_tokens = 0

    def _train_step(self, sae_in: torch.Tensor):
        self.optimizer.zero_grad()
        self.sae.train()
        self.sae.set_decoder_norm_to_unit_norm()

        # Log and reset sparsity stats periodically
        if (self.n_training_steps + 1) % self.cfg.feature_sampling_window == 0:
            if self.cfg.log_to_wandb:
                sparsity_log_dict = self._build_sparsity_log_dict()
                wandb.log(sparsity_log_dict, step=self.n_training_steps)
            self._reset_running_sparsity_stats()

        ghost_grad_neuron_mask = (
            self.n_forward_passes_since_fired > self.cfg.dead_feature_window
        ).bool()
        sae_out, feature_acts, loss_dict = self.sae(sae_in, ghost_grad_neuron_mask)

        with torch.no_grad():
            # Track which features fired
            if self.cfg.class_token:
                did_fire = (feature_acts > 0).float().sum(-2) > 0
                self.act_freq_scores += (feature_acts.abs() > 0).float().sum(0)
            else:
                # PatchSAE-style: activations are [B, N_tokens, d_sae]
                did_fire = (((feature_acts > 0).float().sum(-2) > 0).sum(-2) if feature_acts.dim() > 2 else (feature_acts > 0).float().sum(0)) > 0
                if feature_acts.dim() == 3:
                    self.act_freq_scores += (feature_acts.abs() > 0).float().sum(0).sum(0)
                else:
                    self.act_freq_scores += (feature_acts.abs() > 0).float().sum(0)

            self.n_forward_passes_since_fired += 1
            self.n_forward_passes_since_fired[did_fire] = 0
            self.n_frac_active_tokens += sae_out.size(0)

        self.ghost_grad_neuron_mask = ghost_grad_neuron_mask

        loss_dict["loss"].backward()
        self.sae.remove_gradient_parallel_to_decoder_directions()

        self.optimizer.step()
        self.scheduler.step()

        return sae_out, feature_acts, loss_dict

    def _calculate_sparsity_metrics(self) -> Dict[str, float]:
        feature_freq = self.act_freq_scores / max(self.n_frac_active_tokens, 1)
        return {
            "sparsity/mean_passes_since_fired": self.n_forward_passes_since_fired.mean().item(),
            "sparsity/n_passes_since_fired_over_threshold": (
                self.ghost_grad_neuron_mask.sum().item()
                if self.ghost_grad_neuron_mask is not None else 0
            ),
            "sparsity/below_1e-5": (feature_freq < 1e-5).float().mean().item(),
            "sparsity/below_1e-6": (feature_freq < 1e-6).float().mean().item(),
            "sparsity/dead_features": (
                (feature_freq < self.cfg.dead_feature_threshold).float().mean().item()
            ),
        }

    @torch.no_grad()
    def _log_train_step(
        self,
        feature_acts: torch.Tensor,
        loss_dict: Dict[str, torch.Tensor],
        sae_out: torch.Tensor,
        sae_in: torch.Tensor,
    ):
        metrics = self._calculate_metrics(feature_acts, sae_out, sae_in)
        sparsity_metrics = self._calculate_sparsity_metrics()

        log_dict = {
            "losses/overall_loss": loss_dict["loss"].item(),
            "losses/mse_loss": loss_dict["mse_loss"].item(),
            "losses/l1_loss": loss_dict["l1_loss"].item(),
            "losses/ghost_grad_loss": loss_dict["mse_loss_ghost_resid"].item(),
            **metrics,
            **sparsity_metrics,
            "details/n_training_tokens": self.n_training_tokens,
            "details/current_learning_rate": self.optimizer.param_groups[0]["lr"],
        }
        wandb.log(log_dict, step=self.n_training_steps)

    @torch.no_grad()
    def _calculate_metrics(
        self, feature_acts: torch.Tensor, sae_out: torch.Tensor, sae_in: torch.Tensor,
    ) -> Dict[str, float]:
        if self.cfg.class_token:
            l0 = (feature_acts > 0).float().sum(-1).mean()
        else:
            l0 = (feature_acts > 0).float().sum(-1).mean()

        per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).mean().squeeze()
        total_variance = sae_in.pow(2).sum(-1).mean()
        explained_variance = 1 - per_token_l2_loss / total_variance

        return {
            "metrics/explained_variance": explained_variance.mean().item(),
            "metrics/l0": l0.item(),
        }

    @torch.no_grad()
    def _update_pbar(self, loss_dict, pbar, batch_size):
        pbar.set_description(
            f"{self.n_training_steps}| "
            f"MSE {loss_dict['mse_loss'].item():.4f} | "
            f"L1 {loss_dict['l1_loss'].item():.4f} | "
            f"ExplVar {1 - loss_dict['mse_loss'].item():.3f}"
        )
        pbar.update(batch_size)

    @torch.no_grad()
    def _checkpoint_if_needed(self):
        if (
            self.checkpoint_thresholds
            and self.n_training_tokens > self.checkpoint_thresholds[0]
        ):
            self.save_checkpoint()
            self.checkpoint_thresholds.pop(0)

    def save_checkpoint(self, is_final=False):
        if is_final:
            path = f"{self.cfg.checkpoint_path}/final_{self.sae.get_name()}.pt"
        else:
            path = f"{self.cfg.checkpoint_path}/{self.n_training_tokens}_{self.sae.get_name()}.pt"
        self.sae.save_model(path)

    def fit(self) -> SparseAutoencoder:
        """Main training loop."""
        pbar = tqdm(total=self.cfg.total_training_tokens, desc="Training EEG SAE")

        try:
            while self.n_training_tokens < self.cfg.total_training_tokens:
                sae_acts = self.activation_store.get_batch_activations()
                self.n_training_tokens += sae_acts.size(0)

                sae_out, feature_acts, loss_dict = self._train_step(sae_in=sae_acts)

                if (
                    self.cfg.log_to_wandb
                    and (self.n_training_steps + 1) % self.cfg.wandb_log_frequency == 0
                ):
                    self._log_train_step(
                        feature_acts=feature_acts,
                        loss_dict=loss_dict,
                        sae_out=sae_out,
                        sae_in=sae_acts,
                    )

                self._checkpoint_if_needed()
                self.n_training_steps += 1
                self._update_pbar(loss_dict, pbar, sae_out.size(0))
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        finally:
            print("Saving final checkpoint...")
            self.save_checkpoint(is_final=True)

        pbar.close()
        return self.sae

"""
Sparse Autoencoder for EEG Transformer activations.

Adapted from PatchSAE (https://github.com/dynamical-inference/patchsae) for EEG data.
The SAE learns a sparse overcomplete dictionary of features from REVE model activations.
"""

import gzip
import os
import pickle

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule, HookPoint

from src.sae_training.config import EEGSAERunnerConfig


class SparseAutoencoder(HookedRootModule):
    """Sparse Autoencoder for EEG transformer activations.

    Learns a sparse overcomplete dictionary of features from the residual stream
    (or MLP/attention outputs) of REVE. Supports both standard and gated variants,
    ghost gradients for dead neuron revival, and per-patch (non-class-token) operation.
    """

    def __init__(self, cfg: EEGSAERunnerConfig, device: str):
        super().__init__()
        self.cfg = cfg
        self.d_in = cfg.d_in
        if not isinstance(self.d_in, int):
            raise ValueError(f"d_in must be int, got {self.d_in} ({type(self.d_in)})")
        self.d_sae = cfg.d_sae
        self.l1_coefficient = cfg.l1_coefficient
        self.dtype = cfg.dtype
        self.device = device

        # Weights: W_enc [d_in, d_sae], b_enc [d_sae], W_dec [d_sae, d_in], b_dec [d_in]
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_in, self.d_sae, dtype=self.dtype, device=self.device)
            )
        )
        self.b_enc = nn.Parameter(
            torch.zeros(self.d_sae, dtype=self.dtype, device=self.device)
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_sae, self.d_in, dtype=self.dtype, device=self.device)
            )
        )

        with torch.no_grad():
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

        self.b_dec = nn.Parameter(
            torch.zeros(self.d_in, dtype=self.dtype, device=self.device)
        )

        self.hook_sae_in = HookPoint()
        self.hook_hidden_pre = HookPoint()
        self.hook_hidden_post = HookPoint()
        self.hook_sae_out = HookPoint()

        self.setup()

    def forward(self, x, dead_neuron_mask=None):
        """Forward pass through the SAE.

        Args:
            x: Input activations [..., d_in]
            dead_neuron_mask: Boolean mask of dead neurons [d_sae]

        Returns:
            sae_out: Reconstructed activations
            feature_acts: Sparse feature activations
            loss_dict: Dictionary of loss components
        """
        x = x.to(self.dtype)

        sae_in = self.hook_sae_in(x - self.b_dec)

        hidden_pre = self.hook_hidden_pre(
            einops.einsum(sae_in, self.W_enc, "... d_in, d_in d_sae -> ... d_sae")
            + self.b_enc
        )
        feature_acts = self.hook_hidden_post(F.relu(hidden_pre))

        sae_out = self.hook_sae_out(
            einops.einsum(feature_acts, self.W_dec, "... d_sae, d_sae d_in -> ... d_in")
            + self.b_dec
        )

        # Normalized MSE loss
        mse_loss = (
            torch.pow((sae_out - x.float()), 2)
            / (x**2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        )

        # Ghost gradients for dead neurons
        mse_loss_ghost_resid = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        if (
            self.cfg.use_ghost_grads
            and self.training
            and dead_neuron_mask is not None
            and dead_neuron_mask.sum() > 0
        ):
            residual = x - sae_out
            l2_norm_residual = torch.norm(residual, dim=-1)

            if len(hidden_pre.size()) == 3:
                feature_acts_dead = torch.exp(hidden_pre[:, :, dead_neuron_mask])
            else:
                feature_acts_dead = torch.exp(hidden_pre[:, dead_neuron_mask])

            ghost_out = feature_acts_dead @ self.W_dec[dead_neuron_mask, :]
            l2_norm_ghost_out = torch.norm(ghost_out, dim=-1)
            norm_scaling_factor = l2_norm_residual / (1e-6 + l2_norm_ghost_out * 2)

            if len(hidden_pre.size()) == 3:
                ghost_out = ghost_out * norm_scaling_factor[:, :, None].detach()
            else:
                ghost_out = ghost_out * norm_scaling_factor[:, None].detach()

            mse_loss_ghost_resid = (
                torch.pow((ghost_out - residual.detach().float()), 2)
                / (residual.detach() ** 2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
            )
            mse_rescaling_factor = (mse_loss / (mse_loss_ghost_resid + 1e-6)).detach()
            mse_loss_ghost_resid = mse_rescaling_factor * mse_loss_ghost_resid

        mse_loss_ghost_resid = mse_loss_ghost_resid.mean()
        mse_loss = mse_loss.mean()
        sparsity = torch.abs(feature_acts).sum(dim=-1).mean(dim=(0,))
        l1_loss = self.l1_coefficient * sparsity
        loss = mse_loss + l1_loss + mse_loss_ghost_resid

        loss_dict = {
            "mse_loss": mse_loss,
            "l1_loss": l1_loss.mean(),
            "mse_loss_ghost_resid": mse_loss_ghost_resid,
            "loss": loss.mean(),
        }

        return sae_out, feature_acts, loss_dict

    def encode(self, x):
        """Encode input to sparse feature activations.

        Args:
            x: [..., d_in]
        Returns:
            feature_acts: [..., d_sae]
        """
        x = x.to(self.dtype)
        sae_in = x - self.b_dec
        hidden_pre = einops.einsum(sae_in, self.W_enc, "... d_in, d_in d_sae -> ... d_sae") + self.b_enc
        return F.relu(hidden_pre)

    def decode(self, feature_acts):
        """Decode sparse features back to activation space.

        Args:
            feature_acts: [..., d_sae]
        Returns:
            reconstructed: [..., d_in]
        """
        return einops.einsum(feature_acts, self.W_dec, "... d_sae, d_sae d_in -> ... d_in") + self.b_dec

    @torch.no_grad()
    def initialize_b_dec(self, activation_store):
        """Initialize decoder bias with geometric median or mean of activations."""
        if self.cfg.b_dec_init_method == "geometric_median":
            self._initialize_b_dec_geometric_median(activation_store)
        elif self.cfg.b_dec_init_method == "mean":
            self._initialize_b_dec_mean(activation_store)
        elif self.cfg.b_dec_init_method == "zeros":
            pass
        else:
            raise ValueError(f"Unknown b_dec_init_method: {self.cfg.b_dec_init_method}")

    @torch.no_grad()
    def _initialize_b_dec_geometric_median(self, activation_store, maxiter=100):
        from geom_median.torch import compute_geometric_median

        all_activations = activation_store.get_batch_activations().detach().cpu()
        # Flatten to [N, d_in] if multi-dimensional
        if all_activations.dim() > 2:
            all_activations = all_activations.reshape(-1, all_activations.shape[-1])

        out = compute_geometric_median(
            all_activations, skip_typechecks=True, maxiter=maxiter, per_component=False
        ).median

        if len(out.shape) == 2:
            out = out.mean(dim=0)

        print(f"Reinitializing b_dec with geometric median (shape: {out.shape})")
        self.b_dec.data = out.to(self.dtype).to(self.device)

    @torch.no_grad()
    def _initialize_b_dec_mean(self, activation_store):
        all_activations = activation_store.get_batch_activations().detach().cpu()
        if all_activations.dim() > 2:
            all_activations = all_activations.reshape(-1, all_activations.shape[-1])
        out = all_activations.mean(dim=0)
        print(f"Reinitializing b_dec with mean (shape: {out.shape})")
        self.b_dec.data = out.to(self.dtype).to(self.device)

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """Remove gradient components parallel to decoder directions."""
        parallel_component = einops.einsum(
            self.W_dec.grad, self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component, self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )

    def save_model(self, path: str):
        """Save model state dict and config."""
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)

        state_dict = {"cfg": self.cfg, "state_dict": self.state_dict()}

        if path.endswith(".pt"):
            torch.save(state_dict, path)
        elif path.endswith(".pkl.gz"):
            with gzip.open(path, "wb") as f:
                pickle.dump(state_dict, f)
        else:
            raise ValueError(f"Unsupported extension: {path} (use .pt or .pkl.gz)")

        print(f"Saved SAE to {path}")

    @classmethod
    def load_from_pretrained(cls, path: str, device: str = "cpu"):
        """Load a pretrained SAE from a checkpoint file."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No file found at: {path}")

        if path.endswith(".pt"):
            state_dict = torch.load(path, map_location=device)
        elif path.endswith(".pkl.gz"):
            with gzip.open(path, "rb") as f:
                state_dict = pickle.load(f)
        else:
            raise ValueError(f"Unsupported extension: {path}")

        if "cfg" not in state_dict or "state_dict" not in state_dict:
            raise ValueError("Checkpoint must contain 'cfg' and 'state_dict' keys")

        instance = cls(cfg=state_dict["cfg"], device=device)
        instance.load_state_dict(state_dict["state_dict"])
        return instance

    def get_name(self):
        model_tag = self.cfg.model_name.replace("/", "_")
        return f"eeg_sae_{model_tag}_L{self.cfg.block_layer}_{self.cfg.module_name}_{self.d_sae}"

"""
Activation store for EEG data.

Streams EEG trials from an LMDB dataset, passes them through the REVE model,
and provides batches of intermediate activations for SAE training.

Adapted from PatchSAE's ViTActivationsStore for EEG data.
"""

from __future__ import annotations

import pickle
import random
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.sae_training.hooked_eeg_transformer import HookedEEGTransformer


class SimpleEEGDataset(Dataset):
    """Simple EEG dataset that loads from LMDB.

    Expects LMDB with keys "__keys__" (list of sample keys) and each sample
    stored as a dict with 'sample' (np.ndarray [C, T]) and 'label' (int).
    """

    def __init__(self, db_path: str, keys: Sequence[str], scale_div: float = 100.0):
        super().__init__()
        self.db_path = db_path
        self.keys = list(keys)
        self.scale_div = float(scale_div)
        self._env = None

    def _ensure_env(self):
        if self._env is None:
            import lmdb
            self._env = lmdb.open(
                self.db_path, readonly=True, lock=False,
                readahead=True, meminit=False,
            )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        self._ensure_env()
        k = self.keys[idx]
        with self._env.begin(write=False) as txn:
            pair = pickle.loads(txn.get(k.encode()))
        x = pair["sample"].astype(np.float32) / self.scale_div  # [C, T]
        y = int(pair["label"])
        return torch.from_numpy(x), y

    @staticmethod
    def collate(batch):
        xs = torch.stack([b[0] for b in batch], dim=0)  # [B, C, T]
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return xs, ys


class NumpyEEGDataset(Dataset):
    """EEG dataset from numpy arrays (for non-LMDB data sources)."""

    def __init__(self, data: np.ndarray, labels: np.ndarray, scale_div: float = 1.0):
        """
        Args:
            data: [N, C, T] array of EEG trials
            labels: [N] array of labels
            scale_div: scaling factor
        """
        self.data = data.astype(np.float32) / scale_div
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), int(self.labels[idx])

    @staticmethod
    def collate(batch):
        xs = torch.stack([b[0] for b in batch], dim=0)
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return xs, ys


def load_lmdb_keys(db_path: str) -> List[str]:
    """Load all sample keys from an LMDB database."""
    import lmdb
    env = lmdb.open(db_path, readonly=True, lock=False, readahead=True, meminit=False)
    try:
        with env.begin(write=False) as txn:
            raw = pickle.loads(txn.get(b"__keys__"))
    finally:
        env.close()

    if isinstance(raw, dict):
        out, seen = [], set()
        for mode in ("train", "val", "test"):
            for k in raw.get(mode, []):
                if k not in seen:
                    out.append(k)
                    seen.add(k)
        for mode, lst in raw.items():
            if mode in {"train", "val", "test"}:
                continue
            for k in lst:
                if k not in seen:
                    out.append(k)
                    seen.add(k)
        return out
    return list(raw)


class EEGActivationsStore:
    """
    Streams EEG trials and generates activations from a REVE model.

    Usage:
        store = EEGActivationsStore(cfg, model)
        batch = store.get_batch_activations()  # [B, N_tokens, d_in]
    """

    def __init__(
        self,
        cfg,
        model: HookedEEGTransformer,
        dataset: Optional[Dataset] = None,
        db_path: Optional[str] = None,
    ):
        self.cfg = cfg
        self.model = model
        self.device = str(cfg.device)
        self.batch_size = cfg.batch_size
        self.block_layer = cfg.block_layer
        self.module_name = cfg.module_name
        self.class_token = getattr(cfg, "class_token", False)

        # Build dataset
        if dataset is not None:
            self.dataset = dataset
        elif db_path is not None or cfg.dataset_path:
            path = db_path or cfg.dataset_path
            keys = load_lmdb_keys(path)
            rng = random.Random(cfg.seed)
            rng.shuffle(keys)
            self.dataset = SimpleEEGDataset(path, keys, scale_div=cfg.scale_div)
        else:
            raise ValueError("Must provide either dataset or db_path (or cfg.dataset_path)")

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=getattr(self.dataset, "collate", None),
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        self._iter = iter(self.dataloader)

    def _get_batch_eeg(self):
        """Get a batch of raw EEG data."""
        try:
            batch = next(self._iter)
        except StopIteration:
            self._iter = iter(self.dataloader)
            batch = next(self._iter)

        if isinstance(batch, (tuple, list)):
            eeg = batch[0]
        else:
            eeg = batch
        return eeg.to(self.device)

    @torch.no_grad()
    def get_batch_activations(self) -> torch.Tensor:
        """Get a batch of model activations.

        Returns:
            activations: [B, N_tokens, d_in] where N_tokens = C * H
        """
        eeg = self._get_batch_eeg()
        activations = self.model.get_activations(
            eeg, self.block_layer, self.module_name,
        )
        return activations

    @torch.no_grad()
    def get_batch_eeg_and_activations(self):
        """Get both raw EEG and activations.

        Returns:
            eeg: [B, C, T]
            activations: [B, N_tokens, d_in]
        """
        eeg = self._get_batch_eeg()
        activations = self.model.get_activations(
            eeg, self.block_layer, self.module_name,
        )
        return eeg, activations

    @torch.no_grad()
    def get_batch_eeg_labels_activations(self):
        """Get raw EEG, labels, and activations.

        Returns:
            eeg: [B, C, T]
            labels: [B]
            activations: [B, N_tokens, d_in]
        """
        try:
            batch = next(self._iter)
        except StopIteration:
            self._iter = iter(self.dataloader)
            batch = next(self._iter)

        eeg, labels = batch[0].to(self.device), batch[1].to(self.device)
        activations = self.model.get_activations(
            eeg, self.block_layer, self.module_name,
        )
        return eeg, labels, activations

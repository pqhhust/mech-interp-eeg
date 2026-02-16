"""
BCIC2a Dataset loader for SAE feature evaluation.

Loads 4-class motor imagery EEG from LMDB, compatible with the STELAR
preprocessing pipeline. Provides train/val/test splits by subject.
"""

from __future__ import annotations

import pickle
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class LMDBDataset(Dataset):
    """BCIC2a LMDB dataset: sample [C, T], label int. Scales by /scale_div."""

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
        x = pair["sample"].astype(np.float32) / self.scale_div
        y = int(pair["label"])
        return torch.from_numpy(x), y

    @staticmethod
    def collate(batch):
        xs = torch.stack([b[0] for b in batch], dim=0)
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return xs, ys


def _load_flat_keys(db_path: str) -> List[str]:
    import lmdb
    env = lmdb.open(db_path, readonly=True, lock=False, readahead=True, meminit=False)
    try:
        with env.begin(write=False) as txn:
            raw = pickle.loads(txn.get(b"__keys__"))
    finally:
        env.close()
    if isinstance(raw, dict):
        out, seen = [], set()
        for m in ("train", "val", "test"):
            for k in raw.get(m, []):
                if k not in seen:
                    out.append(k)
                    seen.add(k)
        return out
    return list(raw)


def _key_to_subject(key: str) -> str:
    return key.split("-", 1)[0][:3]


def _group_by_subject(keys: Sequence[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    tokens, seen, by = [], set(), {}
    for k in keys:
        s = _key_to_subject(k)
        if s not in seen:
            tokens.append(s)
            seen.add(s)
        by.setdefault(s, []).append(k)
    return tokens, by


def _normalize_ids(ids: Optional[Sequence[int]], n_subj: int) -> List[int]:
    if not ids:
        return []
    ids = list(dict.fromkeys(int(i) for i in ids))
    if all(1 <= i <= n_subj for i in ids) and not any(i == 0 for i in ids):
        ids = [i - 1 for i in ids]
    return sorted(ids)


def _gather(by_subj, tokens, idxs):
    out = []
    for i in idxs:
        out.extend(by_subj[tokens[i]])
    return out


def get_bcic2a_dataloaders(
    db_path: str,
    batch_size: int = 64,
    test_subject_ids: Optional[List[int]] = None,
    val_subject_ids: Optional[List[int]] = None,
    train_subject_ids: Optional[List[int]] = None,
    scale_div: float = 100.0,
    num_workers: int = 0,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """Create train/val/test DataLoaders for BCIC2a by subject split.

    Args:
        db_path: Path to LMDB database
        batch_size: Batch size
        test_subject_ids: Subject indices for test (1-based)
        val_subject_ids: Subject indices for validation (1-based)
        train_subject_ids: Subject indices for training (1-based); if None, uses all remaining
        scale_div: Scale factor for raw EEG
        num_workers: DataLoader workers
        seed: Random seed

    Returns:
        Dict with "train", "val", "test" DataLoaders
    """
    flat = _load_flat_keys(db_path)
    subj_tokens, by_subj = _group_by_subject(flat)
    n_subj = len(subj_tokens)

    print(f"[BCIC2a] Found {n_subj} subjects: {subj_tokens}")

    te_ids = _normalize_ids(test_subject_ids, n_subj)
    va_ids = _normalize_ids(val_subject_ids, n_subj)
    tr_ids = _normalize_ids(train_subject_ids, n_subj)

    if not tr_ids:
        listed = set(va_ids) | set(te_ids)
        tr_ids = [i for i in range(n_subj) if i not in listed]

    train_keys = _gather(by_subj, subj_tokens, tr_ids)
    val_keys = _gather(by_subj, subj_tokens, va_ids)
    test_keys = _gather(by_subj, subj_tokens, te_ids)

    print(f"[BCIC2a] train={len(train_keys)} | val={len(val_keys)} | test={len(test_keys)}")

    def _make_loader(keys, shuffle):
        if not keys:
            return None
        ds = LMDBDataset(db_path, keys, scale_div)
        return DataLoader(
            ds, batch_size=batch_size, collate_fn=LMDBDataset.collate,
            shuffle=shuffle, num_workers=num_workers, pin_memory=True,
            drop_last=False,
        )

    loaders = {
        "train": _make_loader(train_keys, shuffle=True),
        "test": _make_loader(test_keys, shuffle=False),
    }
    if val_keys:
        loaders["val"] = _make_loader(val_keys, shuffle=False)
    return loaders

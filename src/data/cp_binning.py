import pickle
import h5py
import numpy as np
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class CpBinningStats:
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    num_bins: int

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump({
                "bin_edges": self.bin_edges,
                "bin_centers": self.bin_centers,
                "num_bins": self.num_bins,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "CpBinningStats":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(**data)


def compute_cp_bin_edges(
    h5_paths: List[Path],
    indices: List[int],
    pressure_key: str = "mesh.PressureCoeff",
    num_bins: int = 64,
    clip_percentile: float = 0.1,
) -> CpBinningStats:
    all_cp = []
    for h5_path in h5_paths:
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, "r") as f:
            if pressure_key in f:
                cp = f[pressure_key][...].astype(np.float32).flatten()
                all_cp.append(cp)

    if not all_cp:
        raise ValueError("No Cp data found")

    cp_flat = np.concatenate(all_cp)
    lo = np.percentile(cp_flat, clip_percentile)
    hi = np.percentile(cp_flat, 100.0 - clip_percentile)
    cp_clipped = np.clip(cp_flat, lo, hi)

    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(cp_clipped, quantiles).astype(np.float32)
    bin_edges = np.unique(bin_edges)

    while len(bin_edges) < num_bins + 1:
        mid = (len(bin_edges) - 1) // 2
        insert_val = (bin_edges[mid] + bin_edges[mid + 1]) / 2
        bin_edges = np.insert(bin_edges, mid + 1, insert_val)
    bin_edges = bin_edges[:num_bins + 1]

    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).astype(np.float32)

    return CpBinningStats(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        num_bins=len(bin_centers),
    )


def cp_to_bin_index(cp: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    cp_flat = np.asarray(cp, dtype=np.float64).flatten()
    idx = np.searchsorted(bin_edges[1:-1], cp_flat, side="right")
    return idx.astype(np.int64)


def get_binning_stats_path(num_bins: int, base_dir: str = "./data") -> Path:
    return Path(base_dir) / f"cp_binning_stats_K{num_bins}.pkl"


def load_or_compute_binning_stats(
    h5_paths: List[Path],
    indices: List[int],
    pressure_key: str = "mesh.PressureCoeff",
    num_bins: int = 64,
    base_dir: str = "./data",
    recompute: bool = False,
) -> CpBinningStats:
    stats_path = get_binning_stats_path(num_bins, base_dir)
    if stats_path.exists() and not recompute:
        return CpBinningStats.load(stats_path)
    stats = compute_cp_bin_edges(h5_paths, indices, pressure_key, num_bins)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats.save(stats_path)
    return stats

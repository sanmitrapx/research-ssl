"""Evaluate a DrivAerNet-trained Cp classifier on JakubNetCar (OOD).

Loads the best checkpoint, runs inference on 10 randomly chosen baseline
car shapes from JakubNetCar, and reports per-sample and average rL2.

Usage:
    python scripts/eval_ood_jakubnet.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import pickle
import h5py
import igl
import numpy as np
import torch
from pathlib import Path
from sonata.transform import Compose

from src.data.cfd_h5datamodule_v0 import NormalizationStats

JAKUB_DIR = Path(
    "/mnt/storage02/workspace/research/zephyr/datasets/JakubNetCar/"
    "surface_decimation/"
    "57cdf6b7c130dc35948b0ed04c2af16f2d3410ce733b207255c918dfa59695f0"
)

CKPT_PATH = (
    "/home/sanmitra/research-ssl/outputs/lora_128_ce_emd/"
    "2026-03-12_00-53-58/logs/sonata_cp_classifier/doqo8bw6/"
    "checkpoints/sonata-cp-epoch=134-val_rl2=0.2493.ckpt"
)

GRID_SIZE = 0.01
NUM_SAMPLES = 10
SEED = 42
FREESTREAM = np.array([1.0, 0.0, 0.0], dtype=np.float32)


def robust_normalize_01(arr, lo_pct=1.0, hi_pct=99.0):
    lo = np.percentile(arr, lo_pct)
    hi = np.percentile(arr, hi_pct)
    clipped = np.clip(arr, lo, hi)
    rng = hi - lo
    if rng < 1e-12:
        return np.full_like(arr, 0.5)
    return (clipped - lo) / rng


def compute_physics_color(normals, deltas):
    N = normals.shape[0]
    raw_curv = np.sum(deltas * normals, axis=-1, keepdims=True)
    raw_flow = (normals @ FREESTREAM).reshape(-1, 1)
    raw_lap = np.linalg.norm(deltas, axis=-1, keepdims=True)
    signed_curv = robust_normalize_01(raw_curv)
    flow_angle = (raw_flow + 1.0) / 2.0
    lap_mag = robust_normalize_01(raw_lap)
    return np.concatenate([signed_curv, flow_angle, lap_mag], axis=-1).astype(np.float32)


def build_sonata_transform(grid_size):
    cfg = [
        dict(type="CenterShift", apply_z=True),
        dict(
            type="GridSample",
            grid_size=grid_size,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
            return_inverse=True,
        ),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "inverse"),
            feat_keys=("coord", "color", "normal"),
        ),
    ]
    return Compose(cfg)


def collate_single(point):
    """Add offset tensor for a single-sample batch."""
    point["offset"] = torch.tensor(
        [point["coord"].shape[0]], dtype=torch.long
    )
    return point


def load_sample(h5_path, norm_stats):
    """Load and preprocess a single mesh from JakubNetCar."""
    with h5py.File(h5_path, "r") as f:
        verts = f["mesh.verts"][...].astype(np.float32)
        faces = f["mesh.faces"][...].astype(np.int64)
        pressure = f["mesh.PressureCoeff"][...].astype(np.float32).flatten()
        normals = f["mesh.verts_normals"][...].astype(np.float32)

    laplacian = igl.cotmatrix(verts, faces)
    deltas = laplacian.dot(verts).astype(np.float32)
    # Some decimated meshes have degenerate triangles causing NaN in cotangent weights
    nan_mask = np.isnan(deltas).any(axis=-1)
    if nan_mask.any():
        deltas[nan_mask] = 0.0
    color = compute_physics_color(normals.copy(), deltas.copy())

    if norm_stats is not None:
        local_centroid = (verts.min(axis=0) + verts.max(axis=0)) / 2
        verts_norm = verts - local_centroid
        global_range = np.max(norm_stats.spatial_max - norm_stats.spatial_min)
        verts_norm = verts_norm / (global_range / 2)
    else:
        verts_norm = verts

    data = {"coord": verts_norm, "color": color, "normal": normals}
    extras = {
        "uncentered_coord": torch.from_numpy(verts),
        "untransformed_normal": torch.from_numpy(normals),
        "untransformed_deltas": torch.from_numpy(deltas),
    }
    return data, extras, pressure


def main():
    random.seed(SEED)

    norm_stats = NormalizationStats.load(Path("data/normalisation_stats.pkl"))
    print(f"Loaded normalization stats (pressure_mean={norm_stats.pressure_mean:.4f}, "
          f"pressure_std={norm_stats.pressure_std:.4f})")

    baselines = sorted([
        d.name for d in JAKUB_DIR.iterdir()
        if d.is_dir()
        and d.name.startswith("Baseline_")
        and "Morph" not in d.name
        and "Benchmark" not in d.name
        and (d / "mesh.h5").exists()
    ])
    print(f"Found {len(baselines)} baseline designs in JakubNetCar")

    chosen = random.sample(baselines, NUM_SAMPLES)
    print(f"Selected {NUM_SAMPLES} samples: {chosen}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading model from {CKPT_PATH}")

    from src.models.sonata_cp_lora import SonataCpLoRA
    model = SonataCpLoRA.load_from_checkpoint(CKPT_PATH, map_location=device)
    model.to(device)
    model.eval()
    print(f"Model loaded. Bins: {len(model.bin_centers)}, device: {device}")

    transform = build_sonata_transform(GRID_SIZE)

    rl2_values = []
    print(f"\n{'Design':<35} {'Verts':>8} {'rL2':>10}")
    print("-" * 55)

    for name in chosen:
        h5_path = JAKUB_DIR / name / "mesh.h5"
        data, extras, cp_gt = load_sample(h5_path, norm_stats)

        point = transform(data)
        point.update(extras)
        batch = collate_single(point)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        with torch.no_grad():
            out = model(batch)
        cp_pred = out["cp_hat"].squeeze().cpu().numpy()

        rl2 = np.linalg.norm(cp_pred - cp_gt) / (np.linalg.norm(cp_gt) + 1e-8)
        rl2_values.append(rl2)

        print(f"{name:<35} {len(cp_gt):>8} {rl2:>10.4f}")

    print("-" * 55)
    print(f"{'Average rL2':<35} {'':>8} {np.mean(rl2_values):>10.4f}")
    print(f"{'Std rL2':<35} {'':>8} {np.std(rl2_values):>10.4f}")
    print(f"{'Min rL2':<35} {'':>8} {np.min(rl2_values):>10.4f}")
    print(f"{'Max rL2':<35} {'':>8} {np.max(rl2_values):>10.4f}")


if __name__ == "__main__":
    main()

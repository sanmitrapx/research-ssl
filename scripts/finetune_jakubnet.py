"""Fine-tune LoRA checkpoint on 5 JakubNetCar baselines, eval on 10 others.

Resumes from the best DrivAerNet-trained LoRA checkpoint (val_rl2=0.2493)
and fine-tunes on 5 JakubNetCar baseline car shapes for 50 epochs.
Evaluates on 10 held-out baselines and reports per-sample + average rL2.

Usage:
    python scripts/finetune_jakubnet.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import h5py
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sonata.transform import Compose

from src.data.cfd_h5datamodule_v0 import NormalizationStats
from src.data.cp_binning import cp_to_bin_index

JAKUB_DIR = Path(
    "/mnt/storage02/workspace/research/zephyr/datasets/JakubNetCar/"
    "surface_decimation/"
    "57cdf6b7c130dc35948b0ed04c2af16f2d3410ce733b207255c918dfa59695f0"
)

CKPT_128 = (
    "/home/sanmitra/research-ssl/outputs/lora_128_ce_emd/"
    "2026-03-12_00-53-58/logs/sonata_cp_classifier/doqo8bw6/"
    "checkpoints/sonata-cp-epoch=134-val_rl2=0.2493.ckpt"
)
CKPT_64 = (
    "/home/sanmitra/research-ssl/outputs/lora_64_ce_emd/"
    "2026-03-11_23-31-03/logs/sonata_cp_classifier/1j6d14uy/"
    "checkpoints/sonata-cp-epoch=119-val_rl2=0.2624.ckpt"
)

TRAIN_LIST = Path("data/jakubnet_splits/jakubnet_ft_train.txt")
TEST_LIST = Path("data/jakubnet_splits/jakubnet_ft_test.txt")

EPOCHS = 100
LR = 1e-4
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
    raw_curv = np.sum(deltas * normals, axis=-1, keepdims=True)
    raw_flow = (normals @ FREESTREAM).reshape(-1, 1)
    raw_lap = np.linalg.norm(deltas, axis=-1, keepdims=True)
    signed_curv = robust_normalize_01(raw_curv)
    flow_angle = (raw_flow + 1.0) / 2.0
    lap_mag = robust_normalize_01(raw_lap)
    return np.concatenate([signed_curv, flow_angle, lap_mag], axis=-1).astype(np.float32)


def compute_normals_only_color(normals):
    """Use raw normals (rescaled to [0,1]) as the 3-channel color feature."""
    return ((normals + 1.0) / 2.0).astype(np.float32)


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


def load_and_preprocess(h5_path, norm_stats, use_deltas=True):
    with h5py.File(h5_path, "r") as f:
        verts = f["mesh.verts"][...].astype(np.float32)
        faces = f["mesh.faces"][...].astype(np.int64)
        pressure = f["mesh.PressureCoeff"][...].astype(np.float32).flatten()
        normals = f["mesh.verts_normals"][...].astype(np.float32)

    if use_deltas:
        import igl
        laplacian = igl.cotmatrix(verts, faces)
        deltas = laplacian.dot(verts).astype(np.float32)
        nan_mask = np.isnan(deltas).any(axis=-1)
        if nan_mask.any():
            deltas[nan_mask] = 0.0
        color = compute_physics_color(normals.copy(), deltas.copy())
    else:
        deltas = np.zeros_like(normals)
        color = compute_normals_only_color(normals.copy())

    if norm_stats is not None:
        local_centroid = (verts.min(axis=0) + verts.max(axis=0)) / 2
        verts_norm = verts - local_centroid
        global_range = np.max(norm_stats.spatial_max - norm_stats.spatial_min)
        verts_norm = verts_norm / (global_range / 2)
    else:
        verts_norm = verts

    return {
        "verts": verts,
        "verts_norm": verts_norm,
        "faces": faces,
        "normals": normals,
        "deltas": deltas,
        "color": color,
        "pressure_raw": pressure,
    }


def bin_edges_from_centers(bin_centers):
    """Reconstruct bin edges from bin centers (midpoints between adjacent centers)."""
    bc = np.asarray(bin_centers, dtype=np.float32)
    half_widths = np.diff(bc) / 2.0
    edges = np.empty(len(bc) + 1, dtype=np.float32)
    edges[1:-1] = bc[:-1] + half_widths
    edges[0] = bc[0] - half_widths[0]
    edges[-1] = bc[-1] + half_widths[-1]
    return edges


class JakubNetDataset(Dataset):
    def __init__(self, design_names, jakub_dir, norm_stats, bin_edges, transform,
                 use_deltas=True):
        self.design_names = design_names
        self.jakub_dir = jakub_dir
        self.norm_stats = norm_stats
        self.bin_edges = bin_edges
        self.transform = transform
        self.use_deltas = use_deltas

    def __len__(self):
        return len(self.design_names)

    def __getitem__(self, idx):
        name = self.design_names[idx]
        h5_path = self.jakub_dir / name / "mesh.h5"
        sample = load_and_preprocess(h5_path, self.norm_stats, self.use_deltas)

        data = {
            "coord": sample["verts_norm"],
            "color": sample["color"],
            "normal": sample["normals"],
        }
        point = self.transform(data)

        point["uncentered_coord"] = torch.from_numpy(sample["verts"])
        point["untransformed_normal"] = torch.from_numpy(sample["normals"])
        point["untransformed_deltas"] = torch.from_numpy(sample["deltas"])
        point["pressure_raw"] = torch.from_numpy(sample["pressure_raw"])

        pressure_norm = (sample["pressure_raw"] - self.norm_stats.pressure_mean) / (
            self.norm_stats.pressure_std + 1e-6
        )
        point["pressure"] = torch.from_numpy(pressure_norm.astype(np.float32))

        point["pressure_bin"] = torch.from_numpy(
            cp_to_bin_index(sample["pressure_raw"], self.bin_edges)
            .flatten()
            .astype(np.int64)
        ).long()

        return point


def collate_fn(items):
    collated = {}
    for key in items[0].keys():
        vals = [p[key] for p in items]
        if torch.is_tensor(vals[0]):
            collated[key] = torch.cat(vals)
        else:
            collated[key] = vals[0]

    coord_sizes = [p["coord"].shape[0] for p in items]
    collated["offset"] = torch.cumsum(
        torch.tensor(coord_sizes, dtype=torch.long), dim=0
    )

    if "inverse" in items[0]:
        inv_parts = []
        coord_offset = 0
        for p in items:
            inv_parts.append(p["inverse"] + coord_offset)
            coord_offset += p["coord"].shape[0]
        collated["inverse"] = torch.cat(inv_parts)

    mesh_batch_list = []
    for i, p in enumerate(items):
        n = p["uncentered_coord"].shape[0]
        mesh_batch_list.append(torch.full((n,), i, dtype=torch.long))
    collated["mesh_batch"] = torch.cat(mesh_batch_list)

    return collated


def read_split_file(path):
    names = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                names.append(line.split()[0])
    return names


def evaluate(model, test_names, jakub_dir, norm_stats, transform, device,
             use_deltas=True):
    model.eval()
    rl2_values = []

    print(f"\n{'Design':<35} {'Verts':>8} {'rL2':>10}")
    print("-" * 55)

    for name in test_names:
        h5_path = jakub_dir / name / "mesh.h5"
        sample = load_and_preprocess(h5_path, norm_stats, use_deltas)

        data = {
            "coord": sample["verts_norm"],
            "color": sample["color"],
            "normal": sample["normals"],
        }
        point = transform(data)
        point["uncentered_coord"] = torch.from_numpy(sample["verts"])
        point["untransformed_normal"] = torch.from_numpy(sample["normals"])
        point["untransformed_deltas"] = torch.from_numpy(sample["deltas"])

        point["offset"] = torch.tensor([point["coord"].shape[0]], dtype=torch.long)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in point.items()}

        with torch.no_grad():
            out = model(batch)
        cp_pred = out["cp_hat"].squeeze().cpu().numpy()
        cp_gt = sample["pressure_raw"]

        rl2 = np.linalg.norm(cp_pred - cp_gt) / (np.linalg.norm(cp_gt) + 1e-8)
        rl2_values.append(rl2)

        print(f"{name:<35} {len(cp_gt):>8} {rl2:>10.4f}")

    print("-" * 55)
    print(f"{'Average rL2':<35} {'':>8} {np.mean(rl2_values):>10.4f}")
    print(f"{'Std rL2':<35} {'':>8} {np.std(rl2_values):>10.4f}")
    print(f"{'Min rL2':<35} {'':>8} {np.min(rl2_values):>10.4f}")
    print(f"{'Max rL2':<35} {'':>8} {np.max(rl2_values):>10.4f}")

    return rl2_values


def verify_trainable_params(model):
    """Print a diagnostic summary of which parameter groups are trainable."""
    lora_total = lora_grad = 0
    head_total = head_grad = 0

    for p in model.sonata.parameters():
        lora_total += p.numel()
        if p.requires_grad:
            lora_grad += p.numel()

    for p in model.cp_classifier_head.parameters():
        head_total += p.numel()
        if p.requires_grad:
            head_grad += p.numel()

    print(f"\n--- Parameter verification ---")
    print(f"Sonata (PEFT):  {lora_grad:>10,} / {lora_total:>10,} trainable")
    print(f"Classifier head:{head_grad:>10,} / {head_total:>10,} trainable")

    if head_grad == 0:
        print("WARNING: head has 0 trainable params! Forcing requires_grad=True")
        for p in model.cp_classifier_head.parameters():
            p.requires_grad = True
        head_grad = sum(p.numel() for p in model.cp_classifier_head.parameters())
        print(f"Fixed: head now has {head_grad:,} trainable params")

    if lora_grad == 0:
        print("WARNING: LoRA adapters have 0 trainable params!")

    print(f"-------------------------------\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-deltas", action="store_true",
        help="Skip igl.cotmatrix delta features; use normals as color instead.",
    )
    parser.add_argument("--grid", type=float, default=0.01,
                        help="Grid sampling size (default: 0.01)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint (auto-selects by bins if not given)")
    parser.add_argument("--bins", type=int, default=128, choices=[64, 128],
                        help="Number of bins (selects default ckpt, default: 128)")
    parser.add_argument("--test-split", type=str, default=None,
                        help="Path to test split file (default: jakubnet_ft_test.txt)")
    parser.add_argument("--train-split", type=str, default=None,
                        help="Path to train split file (default: jakubnet_ft_train.txt)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save pre/post rL2 results to this JSON file")
    return parser.parse_args()


def main():
    args = parse_args()
    use_deltas = not args.no_deltas
    grid_size = args.grid

    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        ckpt_path = CKPT_64 if args.bins == 64 else CKPT_128

    pl.seed_everything(42)

    print(f"Config: grid={grid_size}, bins={args.bins}, "
          f"deltas={'ON' if use_deltas else 'OFF'}")
    print(f"Checkpoint: {ckpt_path}")

    norm_stats = NormalizationStats.load(Path("data/normalisation_stats.pkl"))
    print(f"Normalization: pressure_mean={norm_stats.pressure_mean:.4f}, "
          f"pressure_std={norm_stats.pressure_std:.4f}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    bc_tensor = ckpt["state_dict"]["bin_centers"]
    bin_edges = bin_edges_from_centers(bc_tensor.numpy())
    num_bins = len(bc_tensor)
    print(f"Binning: {num_bins} bins (derived from checkpoint)")

    train_split = Path(args.train_split) if args.train_split else TRAIN_LIST
    test_split = Path(args.test_split) if args.test_split else TEST_LIST
    train_names = read_split_file(train_split)
    test_names = read_split_file(test_split)
    print(f"Train: {train_names}")
    print(f"Test:  {test_names}")

    transform = build_sonata_transform(grid_size)

    train_ds = JakubNetDataset(train_names, JAKUB_DIR, norm_stats, bin_edges, transform,
                               use_deltas=use_deltas)
    test_ds = JakubNetDataset(test_names, JAKUB_DIR, norm_stats, bin_edges, transform,
                              use_deltas=use_deltas)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2,
                              collate_fn=collate_fn)
    val_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2,
                            collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading LoRA checkpoint from {ckpt_path}")

    from src.models.sonata_cp_lora import SonataCpLoRA
    model = SonataCpLoRA.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        learning_rate=LR,
    )
    print(f"Model loaded. Bins: {len(model.bin_centers)}")

    # Override optimizer: CosineAnnealingLR instead of OneCycleLR
    from torch.optim import AdamW
    _orig_model = model
    def _cosine_configure_optimizers(self_model):
        lr = LR
        wd = self_model.hparams.weight_decay
        lora_params = [p for p in self_model.sonata.parameters() if p.requires_grad]
        head_params = list(self_model.cp_classifier_head.parameters())
        optimizer = AdamW(
            [{"params": lora_params, "lr": lr},
             {"params": head_params, "lr": lr}],
            weight_decay=wd,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-6
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    import types
    model.configure_optimizers = types.MethodType(_cosine_configure_optimizers, model)
    print("Scheduler: CosineAnnealingLR (smooth)")

    verify_trainable_params(model)

    print("\n=== Pre-finetune evaluation (OOD) ===")
    model.to(device)
    pre_rl2 = evaluate(model, test_names, JAKUB_DIR, norm_stats, transform, device,
                       use_deltas=use_deltas)
    model.cpu()

    print(f"\n=== Fine-tuning for {EPOCHS} epochs on {len(train_names)} samples ===")

    ckpt_dir = Path("outputs/scaling_law/checkpoints")
    if args.output_json:
        tag = Path(args.output_json).stem
        ckpt_dir = Path(args.output_json).parent / f"ckpts_{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    from pytorch_lightning.callbacks import ModelCheckpoint
    ckpt_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best-{epoch:02d}-{val_rl2:.4f}",
        monitor="val_rl2",
        mode="min",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_epochs=EPOCHS,
        check_val_every_n_epoch=5,
        gradient_clip_val=1.0,
        callbacks=[ckpt_callback],
        logger=False,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader, val_loader)

    best_path = ckpt_callback.best_model_path
    best_score = ckpt_callback.best_model_score
    print(f"\nBest checkpoint: {best_path}  (val_rl2={best_score:.4f})")

    # Reload best checkpoint for final eval
    model = SonataCpLoRA.load_from_checkpoint(best_path, map_location="cpu")
    model.to(device)

    print("\n=== Post-finetune evaluation (best ckpt) ===")
    post_rl2 = evaluate(model, test_names, JAKUB_DIR, norm_stats, transform, device,
                        use_deltas=use_deltas)

    if args.output_json:
        import json
        results = {
            "num_train": len(train_names),
            "num_test": len(test_names),
            "grid_size": grid_size,
            "num_bins": num_bins,
            "use_deltas": use_deltas,
            "pre_rl2_mean": float(np.mean(pre_rl2)),
            "pre_rl2_std": float(np.std(pre_rl2)),
            "pre_rl2_per_sample": [float(v) for v in pre_rl2],
            "post_rl2_mean": float(np.mean(post_rl2)),
            "post_rl2_std": float(np.std(post_rl2)),
            "post_rl2_per_sample": [float(v) for v in post_rl2],
            "test_names": test_names,
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

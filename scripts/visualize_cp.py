"""Visualise GT vs Predicted Cp on a 3D car mesh.

Usage:
    python scripts/visualize_cp.py --ckpt <path_to_checkpoint> --sample_idx 0
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import h5py
import numpy as np
import torch
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from src.data.sonata_datamodule import SonataDataModule
from src.models.sonata_cp_classifier import SonataCpClassifier


def load_raw_mesh(h5_path):
    """Load raw vertices, faces, and Cp from an H5 file."""
    with h5py.File(h5_path, "r") as f:
        verts = f["mesh.verts"][...].astype(np.float64)
        faces = f["mesh.faces"][...].astype(np.int64)
        cp = f["mesh.PressureCoeff"][...].astype(np.float64).flatten()
    return verts, faces, cp


def build_pv_mesh(verts, faces):
    """Create a PyVista PolyData from vertices and triangular faces."""
    nf = faces.shape[0]
    cells = np.hstack([np.full((nf, 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(verts, cells)


def _render_panel(mesh, scalars, clim, cmap, title, cam_pos, parallel=False, window_size=(800, 600)):
    """Render a single PyVista panel to an image array."""
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.add_mesh(
        mesh,
        scalars=scalars,
        clim=clim,
        cmap=cmap,
        show_scalar_bar=False,
        smooth_shading=True,
    )
    pl.set_background("white")
    pl.camera_position = cam_pos
    if parallel:
        pl.camera.parallel_projection = True

    pl.reset_camera_clipping_range()
    near, far = pl.camera.clipping_range
    pl.camera.clipping_range = (near * 0.01, far * 10.0)

    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def plot_comparison(verts, faces, cp_gt, cp_pred, rl2, out_path):
    """Create a 2x3 comparison figure (iso + top views for GT, Pred, Diff)."""
    mesh = build_pv_mesh(verts, faces)
    cp_diff = cp_pred - cp_gt

    cx, cy, cz = verts.mean(axis=0)
    L = verts.ptp(axis=0).max()

    cam_iso = [
        (cx + L * 1.2, cy - L * 1.2, cz + L * 0.8),
        (cx, cy, cz),
        (0, 0, 1),
    ]
    cam_top = [
        (cx, cy, cz + L * 1.5),
        (cx, cy, cz),
        (1, 0, 0),
    ]

    cp_lo, cp_hi = np.percentile(cp_gt, [1, 99])
    diff_abs = max(abs(cp_diff.min()), abs(cp_diff.max()), 0.1)

    imgs = {}
    for label, scalars, clim, cmap in [
        ("gt", cp_gt, (cp_lo, cp_hi), "jet"),
        ("pred", cp_pred, (cp_lo, cp_hi), "jet"),
        ("diff", cp_diff, (-diff_abs, diff_abs), "RdBu_r"),
    ]:
        mesh[label] = scalars
        imgs[f"{label}_iso"] = _render_panel(mesh, label, clim, cmap, label, cam_iso)
        imgs[f"{label}_top"] = _render_panel(mesh, label, clim, cmap, label, cam_top, parallel=True)

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 0.08], hspace=0.05, wspace=0.05)

    titles_top = ["Ground Truth", "Prediction", "Difference"]
    for col, key_prefix in enumerate(["gt", "pred", "diff"]):
        ax_iso = fig.add_subplot(gs[0, col])
        ax_iso.imshow(imgs[f"{key_prefix}_iso"])
        ax_iso.set_title(titles_top[col], fontsize=14, fontweight="bold")
        ax_iso.axis("off")

        ax_top = fig.add_subplot(gs[1, col])
        ax_top.imshow(imgs[f"{key_prefix}_top"])
        ax_top.axis("off")

    ax_cb_cp = fig.add_subplot(gs[2, :2])
    norm_cp = Normalize(vmin=cp_lo, vmax=cp_hi)
    sm_cp = ScalarMappable(cmap="jet", norm=norm_cp)
    cb_cp = fig.colorbar(sm_cp, cax=ax_cb_cp, orientation="horizontal")
    cb_cp.set_label("Cp", fontsize=12)

    ax_cb_diff = fig.add_subplot(gs[2, 2])
    norm_diff = Normalize(vmin=-diff_abs, vmax=diff_abs)
    sm_diff = ScalarMappable(cmap="RdBu_r", norm=norm_diff)
    cb_diff = fig.colorbar(sm_diff, cax=ax_cb_diff, orientation="horizontal")
    cb_diff.set_label("ΔCp", fontsize=12)

    fig.suptitle(f"rL2 = {rl2:.4f}", fontsize=16, fontweight="bold", y=0.98)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--sample_idx", type=int, default=0, help="Test sample index")
    parser.add_argument("--out", default="plots/cp_pred.png", help="Output path")
    args = parser.parse_args()

    dm = SonataDataModule(
        data_dir="/mnt/storage01/workspace/research/zephyr/02_processed/DrivAerNet/140b9b2dc4e57ccc4e6856fdc1a116126b724ba2522232f88131827edc9e602e",
        batch_size=1,
        num_workers=0,
        use_cp_binning=True,
        num_cp_bins=64,
        color_mode="physics",
    )

    model = SonataCpClassifier.load_from_checkpoint(args.ckpt, map_location="cpu")
    model.eval()

    test_ds = dm.base.test_dataset
    item = test_ds[args.sample_idx]
    h5_path = test_ds.h5_paths[args.sample_idx]

    raw_verts, raw_faces, raw_cp = load_raw_mesh(h5_path)
    design_name = h5_path.parent.name

    transform = dm._test_transform
    batch = dm._transform_item(item, transform)
    batch = {k: v.unsqueeze(0) if torch.is_tensor(v) and v.dim() >= 1 else v for k, v in batch.items()}

    with torch.no_grad():
        out = model(batch)
    cp_pred = out["cp_hat"].squeeze().numpy()

    rl2 = np.linalg.norm(cp_pred - raw_cp) / (np.linalg.norm(raw_cp) + 1e-8)

    out_path = args.out if args.out != "plots/cp_pred.png" else f"plots/{design_name}_cp_pred.png"
    plot_comparison(raw_verts, raw_faces, raw_cp, cp_pred, rl2, out_path)


if __name__ == "__main__":
    main()

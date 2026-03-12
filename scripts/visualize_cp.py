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
from src.models.resnet_heads.sonata_cp_regression import SonataCpRegression


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


def _render_panel(mesh, scalars, clim, cmap, title, cam_pos, parallel=False, zoom=1.0, window_size=(800, 600)):
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
    if zoom != 1.0:
        pl.camera.zoom(zoom)

    pl.reset_camera_clipping_range()
    near, far = pl.camera.clipping_range
    pl.camera.clipping_range = (near * 0.01, far * 10.0)

    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def plot_comparison(verts, faces, cp_gt, cp_pred, rl2, out_path, style="default"):
    """Create a 2x3 comparison figure (iso + top views for GT, Pred, Diff)."""
    mesh = build_pv_mesh(verts, faces)
    cp_diff = cp_pred - cp_gt

    cx, cy, cz = verts.mean(axis=0)
    L = verts.ptp(axis=0).max()

    cam_iso = [
        (cx - L * 0.9, cy - L * 0.9, cz + L * 0.6),
        (cx, cy, cz),
        (0, 0, 1),
    ]
    cam_top = [
        (cx, cy, cz + L * 1.5),
        (cx, cy, cz),
        (0, -1, 0),
    ]

    if style == "paper":
        cp_clim = (-2.0, 1.0)
        diff_lim = 0.10
        cp_cmap = "RdBu_r"
        cp_cb_label = "GT / Pred"
        diff_cb_label = "Difference"
    else:
        cp_lo, cp_hi = np.percentile(cp_gt, [1, 99])
        cp_clim = (cp_lo, cp_hi)
        diff_lim = max(abs(cp_diff.min()), abs(cp_diff.max()), 0.1)
        cp_cmap = "jet"
        cp_cb_label = "Cp"
        diff_cb_label = "ΔCp"

    imgs = {}
    for label, scalars, clim, cmap in [
        ("gt", cp_gt, cp_clim, cp_cmap),
        ("pred", cp_pred, cp_clim, cp_cmap),
        ("diff", cp_diff, (-diff_lim, diff_lim), "RdBu_r"),
    ]:
        mesh[label] = scalars
        imgs[f"{label}_iso"] = _render_panel(mesh, label, clim, cmap, label, cam_iso)
        imgs[f"{label}_top"] = _render_panel(mesh, label, clim, cmap, label, cam_top, parallel=True, zoom=0.55)

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
    norm_cp = Normalize(vmin=cp_clim[0], vmax=cp_clim[1])
    sm_cp = ScalarMappable(cmap=cp_cmap, norm=norm_cp)
    cb_cp = fig.colorbar(sm_cp, cax=ax_cb_cp, orientation="horizontal")
    cb_cp.set_label(cp_cb_label, fontsize=12)

    ax_cb_diff = fig.add_subplot(gs[2, 2])
    norm_diff = Normalize(vmin=-diff_lim, vmax=diff_lim)
    sm_diff = ScalarMappable(cmap="RdBu_r", norm=norm_diff)
    cb_diff = fig.colorbar(sm_diff, cax=ax_cb_diff, orientation="horizontal")
    cb_diff.set_label(diff_cb_label, fontsize=12)

    if style == "paper":
        fig.text(0.05, 0.02, f"RL2: {rl2:.4f}", fontsize=14, fontweight="bold")
    else:
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
    parser.add_argument("--style", default="default", choices=["default", "paper"],
                        help="Plot style: 'default' (jet/percentile) or 'paper' (RdBu_r/fixed range)")
    args = parser.parse_args()

    dm = SonataDataModule(
        data_dir="/mnt/storage01/workspace/research/zephyr/02_processed/DrivAerNet/140b9b2dc4e57ccc4e6856fdc1a116126b724ba2522232f88131827edc9e602e",
        batch_size=1,
        num_workers=0,
        use_cp_binning=True,
        num_cp_bins=64,
        color_mode="physics",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_data = torch.load(args.ckpt, map_location="cpu")
    is_regression = "regression_head.0.weight" in ckpt_data.get("state_dict", {})

    if is_regression:
        model = SonataCpRegression.load_from_checkpoint(args.ckpt, map_location=device)
    else:
        model = SonataCpClassifier.load_from_checkpoint(args.ckpt, map_location=device)
    model.to(device)
    model.eval()

    test_ds = dm.base.test_dataset
    item = test_ds[args.sample_idx]
    h5_path = test_ds.h5_paths[args.sample_idx]

    raw_verts, raw_faces, raw_cp = load_raw_mesh(h5_path)
    design_name = h5_path.parent.name

    transform = dm._test_transform
    sample = dm._transform_item(item, transform)
    batch = dm._collate_batched([sample])

    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    with torch.no_grad():
        out = model(batch)
    cp_pred = out["cp_hat"].squeeze()
    if is_regression:
        cp_pred = model._denormalize(cp_pred)
    cp_pred = cp_pred.cpu().numpy()

    rl2 = np.linalg.norm(cp_pred - raw_cp) / (np.linalg.norm(raw_cp) + 1e-8)

    out_path = args.out if args.out != "plots/cp_pred.png" else f"plots/{design_name}_cp_pred.png"
    plot_comparison(raw_verts, raw_faces, raw_cp, cp_pred, rl2, out_path, style=args.style)


if __name__ == "__main__":
    main()

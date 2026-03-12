"""Sonata data module: wraps H5DataModule with Sonata transforms and batched collation."""

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from .cfd_h5datamodule_v0 import H5DataModule
from sonata.transform import Compose


class _TransformedDataset(Dataset):
    """Thin wrapper that applies a transform to each item from a base dataset."""

    def __init__(self, base_dataset, transform_fn):
        self.base = base_dataset
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.transform_fn(self.base[idx])


class SonataDataModule(pl.LightningDataModule):
    """Wraps H5DataModule with Sonata-specific transforms and collation.

    Each mesh sample is converted to a Sonata-compatible Point dict with
    grid-sampled coordinates, an inverse mapping from original points to
    grid cells, and original-resolution geometric features needed for
    per-point Cp prediction.
    """

    VALID_COLOR_MODES = ("normals", "physics", "curv_only", "flow_only", "lap_only")

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1,
        num_workers: int = 32,
        h5_filename: str = "mesh.h5",
        train_list_file: str = "train_design_ids.txt",
        test_list_file: str = "test_design_ids.txt",
        normalize: bool = True,
        grid_size: float = 0.02,
        batching_enabled: bool = True,
        use_cp_binning: bool = True,
        num_cp_bins: int = 256,
        recompute_stats: bool = False,
        color_mode: str = "physics",
        freestream_direction: list = None,
    ):
        super().__init__()
        if color_mode not in self.VALID_COLOR_MODES:
            raise ValueError(
                f"color_mode must be one of {self.VALID_COLOR_MODES}, got '{color_mode}'"
            )
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.grid_size = grid_size
        self.batching_enabled = batching_enabled
        self.color_mode = color_mode
        self.freestream_dir = np.array(
            freestream_direction if freestream_direction is not None else (1.0, 0.0, 0.0),
            dtype=np.float32,
        )

        self.base = H5DataModule(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            h5_filename=h5_filename,
            train_list_file=train_list_file,
            test_list_file=test_list_file,
            normalize=normalize,
            use_cp_binning=use_cp_binning,
            num_cp_bins=num_cp_bins,
        )

        self._train_transform = self._build_transform("train")
        self._test_transform = self._build_transform("test")

    @property
    def bin_centers(self):
        return self.base.cp_binning_stats.bin_centers.tolist()

    def _build_transform(self, mode):
        # GridSample test mode returns a list of augmentations which breaks
        # downstream transforms; use train mode for both (only difference is
        # which point within each voxel is selected -- negligible variance).
        cfg = [
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=self.grid_size,
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

    @staticmethod
    def _robust_normalize_01(arr, lo_pct=1.0, hi_pct=99.0):
        """Clip to [p_lo, p_hi] and linearly scale to [0, 1]."""
        lo = np.percentile(arr, lo_pct)
        hi = np.percentile(arr, hi_pct)
        clipped = np.clip(arr, lo, hi)
        rng = hi - lo
        if rng < 1e-12:
            return np.full_like(arr, 0.5)
        return (clipped - lo) / rng

    def _compute_color(self, normals, deltas):
        """Build the 3-channel color input based on ``self.color_mode``.

        All physics channels are scaled to [0, 1] to match the RGB range
        Sonata was pretrained with:
            flow_angle  – deterministic: (dot(n, freestream) + 1) / 2
            signed_curv – robust percentile normalization per sample
            lap_mag     – robust percentile normalization per sample

        Modes:
            normals   → [nx, ny, nz]  (kept as-is, range [-1, 1])
            physics   → [signed_curv, flow_angle, lap_mag]
            curv_only → [signed_curv, 0.5, 0.5]
            flow_only → [0.5, flow_angle, 0.5]
            lap_only  → [0.5, 0.5, lap_mag]
        """
        if self.color_mode == "normals":
            return normals.copy()

        N = normals.shape[0]

        raw_curv = np.sum(deltas * normals, axis=-1, keepdims=True)
        raw_flow = (normals @ self.freestream_dir).reshape(-1, 1)
        raw_lap = np.linalg.norm(deltas, axis=-1, keepdims=True)

        signed_curv = self._robust_normalize_01(raw_curv)
        flow_angle = (raw_flow + 1.0) / 2.0
        lap_mag = self._robust_normalize_01(raw_lap)

        neutral = np.full((N, 1), 0.5, dtype=np.float32)

        channel_map = {
            "physics": (signed_curv, flow_angle, lap_mag),
            "curv_only": (signed_curv, neutral, neutral),
            "flow_only": (neutral, flow_angle, neutral),
            "lap_only": (neutral, neutral, lap_mag),
        }

        c0, c1, c2 = channel_map[self.color_mode]
        return np.concatenate([c0, c1, c2], axis=-1).astype(np.float32)

    def _transform_item(self, item, transform):
        """Convert an H5MeshDataset dict into a Sonata-compatible Point dict."""
        normals = item["normals"].astype(np.float32)
        deltas = item["deltas"].astype(np.float32)
        color = self._compute_color(normals.copy(), deltas.copy())

        data = {
            "coord": item["vertices"].astype(np.float32),
            "color": color,
            "normal": normals,
        }

        extras = {
            "uncentered_coord": torch.from_numpy(item["vertices"].astype(np.float32)),
            "untransformed_normal": torch.from_numpy(normals),
            "untransformed_deltas": torch.from_numpy(deltas),
            "pressure": torch.from_numpy(item["pressure"].flatten().astype(np.float32)),
        }

        if "pressure_raw" in item:
            extras["pressure_raw"] = torch.from_numpy(
                item["pressure_raw"].flatten().astype(np.float32)
            )
        if "pressure_bin" in item:
            extras["pressure_bin"] = torch.from_numpy(
                item["pressure_bin"].flatten().astype(np.int64)
            ).long()

        if "faces" in item:
            extras["faces"] = torch.from_numpy(
                item["faces"].astype(np.int64)
            )

        point = transform(data)
        point.update(extras)
        return point

    def _collate_batched(self, items):
        """Collate multiple transformed items into a single batched dict.

        Crucially:
        - Builds cumulative ``offset`` for Sonata's Point structure
        - Shifts ``inverse`` indices so they index into the concatenated coord
        - Builds ``mesh_batch`` for mapping original-resolution points to samples
        """
        collated = {}

        # Concatenate all tensor keys, keep scalars from first item
        for key in items[0].keys():
            vals = [p[key] for p in items]
            if torch.is_tensor(vals[0]):
                collated[key] = torch.cat(vals)
            else:
                collated[key] = vals[0]

        # Sonata expects offset = cumsum of per-sample grid-sampled coord counts
        coord_sizes = [p["coord"].shape[0] for p in items]
        collated["offset"] = torch.cumsum(
            torch.tensor(coord_sizes, dtype=torch.long), dim=0
        )

        # Shift inverse indices to point into concatenated coord tensor
        if "inverse" in items[0]:
            inv_parts = []
            coord_offset = 0
            for p in items:
                inv_parts.append(p["inverse"] + coord_offset)
                coord_offset += p["coord"].shape[0]
            collated["inverse"] = torch.cat(inv_parts)

        # mesh_batch: which sample each original-resolution point belongs to
        mesh_batch_list = []
        for i, p in enumerate(items):
            n = p["uncentered_coord"].shape[0]
            mesh_batch_list.append(torch.full((n,), i, dtype=torch.long))
        collated["mesh_batch"] = torch.cat(mesh_batch_list)

        # Offset face vertex indices for batched samples
        if "faces" in items[0]:
            face_parts = []
            vertex_offset = 0
            for p in items:
                face_parts.append(p["faces"] + vertex_offset)
                vertex_offset += p["uncentered_coord"].shape[0]
            collated["faces"] = torch.cat(face_parts)

        return collated

    def _get_loader(self, dataset, transform, shuffle):
        wrapped = _TransformedDataset(
            dataset, lambda item: self._transform_item(item, transform)
        )
        return DataLoader(
            wrapped,
            batch_size=self.batch_size if self.batching_enabled else 1,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_batched,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._get_loader(self.base.train_dataset, self._train_transform, shuffle=True)

    def val_dataloader(self):
        return self._get_loader(self.base.test_dataset, self._test_transform, shuffle=False)

    def test_dataloader(self):
        return self._get_loader(self.base.test_dataset, self._test_transform, shuffle=False)

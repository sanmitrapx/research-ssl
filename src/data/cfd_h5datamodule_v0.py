import h5py
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
import igl
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .cp_binning import CpBinningStats, compute_cp_bin_edges, cp_to_bin_index, get_binning_stats_path


@dataclass
class NormalizationStats:
    """Store normalization statistics for consistent transforms"""

    spatial_min: np.ndarray
    spatial_max: np.ndarray
    pressure_mean: float
    pressure_std: float

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: Path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Handle legacy stats files that may contain extra fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**data)


class H5MeshDataset(Dataset):
    """Dataset for loading mesh data from H5 files"""

    def __init__(
        self,
        h5_paths: List[Path],
        indices: List[int],
        norm_stats: Optional[NormalizationStats] = None,
        device: str = "cpu",
        vertices_key: str = "mesh.verts",
        faces_key: str = "mesh.faces",
        pressure_key: str = "mesh.PressureCoeff",
        normals_key: str = "mesh.verts_normals",
        drag: str = "drag_coeff_truth",
        recompute_deltas: bool = True,
        use_cp_binning: bool = False,
        cp_binning_stats: Optional[CpBinningStats] = None,
    ):
        self.h5_paths = h5_paths
        self.indices = indices
        self.norm_stats = norm_stats
        self.device = device
        self.recompute_deltas = recompute_deltas
        self.use_cp_binning = use_cp_binning
        self.cp_binning_stats = cp_binning_stats

        self.vertices_key = vertices_key
        self.faces_key = faces_key
        self.pressure_key = pressure_key
        self.normals_key = normals_key
        self.drag = drag

    def __len__(self):
        return len(self.h5_paths)

    def _load_from_h5(self, h5_path: Path, idx: int) -> Dict[str, np.ndarray]:
        """Load all data from H5 file"""
        data = {}
        if not h5_path.exists():
            return data

        with h5py.File(h5_path, "r") as h5file:
            if self.vertices_key in h5file:
                data["vertices"] = h5file[self.vertices_key][...].astype(np.float32)

            if self.faces_key in h5file:
                data["faces"] = h5file[self.faces_key][...].astype(np.int64)

            if self.pressure_key in h5file:
                data["pressure"] = h5file[self.pressure_key][...].astype(np.float32)

            if self.normals_key in h5file:
                data["normals"] = h5file[self.normals_key][...].astype(np.float32)

            if self.drag in h5file:
                data["drag"] = h5file[self.drag][...]

        return data

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        h5_path = self.h5_paths[idx]
        data_idx = self.indices[idx]

        data = self._load_from_h5(h5_path, data_idx)
        if not data:
            return {}

        # Store raw Cp before normalization (needed for classification targets)
        if "pressure" in data and self.use_cp_binning:
            data["pressure_raw"] = data["pressure"].flatten().copy()

        if self.norm_stats is not None and "vertices" in data:
            data["vertices"] = self._normalize_spatial(data["vertices"])

        if "pressure" in data:
            data["pressure"] = self._normalize_pressure(data["pressure"])

        # Compute bin indices after normalization (uses raw Cp)
        if self.use_cp_binning and "pressure_raw" in data:
            data["pressure_bin"] = cp_to_bin_index(
                data["pressure_raw"], self.cp_binning_stats.bin_edges
            )

        deltas_file = Path("./data/deltas/" + h5_path.parts[-2] + ".pkl")
        if deltas_file.exists() and not self.recompute_deltas:
            with open(deltas_file, "rb") as f:
                data["deltas"] = pickle.load(f)
        else:
            laplacian = igl.cotmatrix(data["vertices"], data["faces"])
            data["deltas"] = laplacian.dot(data["vertices"])
            with open(deltas_file, "wb") as f:
                pickle.dump(data["deltas"], f)

        return data

    def _normalize_spatial(self, vertices: np.ndarray) -> np.ndarray:
        """Center per-sample, scale globally (isotropic)."""
        local_min = vertices.min(axis=0)
        local_max = vertices.max(axis=0)
        local_centroid = (local_max + local_min) / 2

        vertices = vertices - local_centroid

        global_range = np.max(self.norm_stats.spatial_max - self.norm_stats.spatial_min)
        vertices = vertices / (global_range / 2)

        return vertices

    def _normalize_pressure(self, pressure: np.ndarray) -> np.ndarray:
        if self.norm_stats is None:
            return pressure
        mean = self.norm_stats.pressure_mean
        std = self.norm_stats.pressure_std
        return (pressure - mean) / (std + 1e-6)


class H5DataModule:
    """Data module for H5 file loading"""

    def __init__(
        self,
        data_dir: Path,
        device: str = "cpu",
        batch_size: int = 4,
        num_workers: int = 4,
        h5_filename: str = "mesh.h5",
        train_list_file: str = "train_design_ids.txt",
        test_list_file: str = "test_design_ids.txt",
        vertices_key: str = "mesh.verts",
        faces_key: str = "mesh.faces",
        pressure_key: str = "mesh.PressureCoeff",
        normals_key: str = "mesh.verts_normals",
        drag: str = "drag_coeff_truth",
        normalize: bool = True,
        use_cp_binning: bool = False,
        num_cp_bins: int = 64,
    ):
        self.data_dir = Path(data_dir)
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.h5_filename = h5_filename
        self.normalize = normalize
        self.use_cp_binning = use_cp_binning
        self.num_cp_bins = num_cp_bins

        self.keys = {
            "vertices_key": vertices_key,
            "faces_key": faces_key,
            "pressure_key": pressure_key,
            "normals_key": normals_key,
            "drag": drag,
        }

        self.train_dirs, self.train_indices = self._load_design_list(train_list_file)
        self.test_dirs, self.test_indices = self._load_design_list(test_list_file)

        self.train_h5_paths = [d / h5_filename for d in self.train_dirs]
        self.test_h5_paths = [d / h5_filename for d in self.test_dirs]

        self.train_h5_paths, self.train_indices = self._filter_existing_paths(
            self.train_h5_paths, self.train_indices, "training"
        )
        self.test_h5_paths, self.test_indices = self._filter_existing_paths(
            self.test_h5_paths, self.test_indices, "test"
        )

        self.norm_stats = None
        if normalize:
            stats_path = Path("./data/normalisation_stats.pkl")
            if stats_path.exists():
                self.norm_stats = NormalizationStats.load(stats_path)
            else:
                self.norm_stats = self._compute_normalization_stats()
                self.norm_stats.save(stats_path)

        self.cp_binning_stats = None
        if use_cp_binning:
            bin_stats_path = get_binning_stats_path(num_cp_bins)
            if bin_stats_path.exists():
                self.cp_binning_stats = CpBinningStats.load(bin_stats_path)
                print(f"Loaded Cp binning stats from {bin_stats_path}")
            else:
                print("Computing Cp binning stats from training set...")
                self.cp_binning_stats = compute_cp_bin_edges(
                    h5_paths=self.train_h5_paths,
                    indices=self.train_indices,
                    pressure_key=pressure_key,
                    num_bins=num_cp_bins,
                )
                bin_stats_path.parent.mkdir(parents=True, exist_ok=True)
                self.cp_binning_stats.save(bin_stats_path)
                print(f"Saved Cp binning stats to {bin_stats_path}")

        self._train_dataset = H5MeshDataset(
            h5_paths=self.train_h5_paths,
            indices=self.train_indices,
            norm_stats=self.norm_stats,
            device=device,
            use_cp_binning=use_cp_binning,
            cp_binning_stats=self.cp_binning_stats,
            **self.keys,
        )

        self._test_dataset = H5MeshDataset(
            h5_paths=self.test_h5_paths,
            indices=self.test_indices,
            norm_stats=self.norm_stats,
            device=device,
            use_cp_binning=use_cp_binning,
            cp_binning_stats=self.cp_binning_stats,
            **self.keys,
        )

    def _filter_existing_paths(self, h5_paths, indices, split_name):
        valid_paths = []
        valid_indices = []
        for path, idx in zip(h5_paths, indices):
            if path.exists():
                valid_paths.append(path)
                valid_indices.append(idx)
            else:
                print(f"Skipping missing {split_name} file: {path}")

        if not valid_paths:
            raise ValueError(f"No valid H5 files found for {split_name} split")

        print(f"Found {len(valid_paths)}/{len(h5_paths)} valid {split_name} files")
        return valid_paths, valid_indices

    def _load_design_list(self, list_file: str) -> Tuple[List[Path], List[int]]:
        """Load design directories and indices from text file in split directory"""
        split_dir = self.data_dir.parent / "splits_tenpercent"
        list_path = split_dir / list_file

        dirs = []
        indices = []
        with open(list_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    dir_name = parts[0]
                    idx = int(parts[1]) if len(parts) > 1 else 1
                    dirs.append(self.data_dir / dir_name)
                    indices.append(idx)

        return dirs, indices

    def _compute_normalization_stats(self) -> NormalizationStats:
        """Compute normalization statistics using streaming/online algorithms"""
        spatial_min = None
        spatial_max = None

        pressure_n = 0
        pressure_mean = 0.0
        pressure_M2 = 0.0

        for i, (h5_path, idx) in enumerate(zip(self.train_h5_paths, self.train_indices)):
            if not h5_path.exists():
                continue

            with h5py.File(h5_path, "r") as h5file:
                if (i + 1) % 50 == 0:
                    print(f"Computing stats: {i + 1}/{len(self.train_h5_paths)}")

                vk = self.keys["vertices_key"]
                if vk in h5file:
                    vertices = h5file[vk][...]
                    if spatial_min is None:
                        spatial_min = vertices.min(axis=0)
                        spatial_max = vertices.max(axis=0)
                    else:
                        spatial_min = np.minimum(spatial_min, vertices.min(axis=0))
                        spatial_max = np.maximum(spatial_max, vertices.max(axis=0))

                pk = self.keys["pressure_key"]
                if pk in h5file:
                    pressure_data = h5file[pk][...].flatten()
                    for value in pressure_data:
                        pressure_n += 1
                        delta = value - pressure_mean
                        pressure_mean += delta / pressure_n
                        delta2 = value - pressure_mean
                        pressure_M2 += delta * delta2

        pressure_std = np.sqrt(pressure_M2 / pressure_n) if pressure_n > 1 else 0.0

        if spatial_min is None:
            spatial_min = np.array([0.0, 0.0, 0.0])
            spatial_max = np.array([1.0, 1.0, 1.0])

        return NormalizationStats(
            spatial_min=spatial_min,
            spatial_max=spatial_max,
            pressure_mean=pressure_mean,
            pressure_std=pressure_std,
        )

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    def train_dataloader(self, **kwargs):
        default_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "num_workers": self.num_workers,
            "pin_memory": self.device != "cpu",
        }
        default_kwargs.update(kwargs)
        return DataLoader(self._train_dataset, **default_kwargs)

    def test_dataloader(self, **kwargs):
        default_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.device != "cpu",
        }
        default_kwargs.update(kwargs)
        return DataLoader(self._test_dataset, **default_kwargs)

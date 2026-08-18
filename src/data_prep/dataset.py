"""Windowed tensor dataset for graph-based sequence models (e.g. LiteSTGNN).

Turns the processed ERA5 monthly files for one chronological split into
fixed-length (input_window, target_window) tensor pairs, with the lat/lon
grid flattened into a single node dimension. Reuses list_split_files from
split.py rather than duplicating any split logic.
"""

import numpy as np
import torch
import xarray as xr

from src.data_prep.config import PROCESSED_DIR
from src.data_prep.split import list_split_files


def load_variable_grid(files, variable: str, spatial_stride: int = 1):
    """Load one variable across files, concatenated in time.

    Returns a [time, n_nodes] float32 array, plus the (lat, lon) coordinate
    arrays used to flatten the grid into nodes.
    """
    arrays = []
    lat = lon = None
    for path in files:
        ds = xr.open_dataset(path)
        da = ds[variable].isel(
            latitude=slice(None, None, spatial_stride),
            longitude=slice(None, None, spatial_stride),
        )
        if lat is None:
            lat = da["latitude"].values
            lon = da["longitude"].values
        values = da.values.reshape(da.values.shape[0], -1)
        arrays.append(values)
        ds.close()
    data = np.concatenate(arrays, axis=0).astype(np.float32)
    return data, lat, lon


class Scaler:
    """Global mean/std standardization, fit once on training data only."""

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, data: np.ndarray) -> "Scaler":
        return cls(mean=float(data.mean()), std=float(data.std()) + 1e-6)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


def make_windows(data: np.ndarray, seq_len: int, pred_len: int):
    """Slice a [time, n_nodes] array into overlapping (input, target) windows."""
    n_windows = len(data) - seq_len - pred_len + 1
    if n_windows <= 0:
        raise ValueError(
            f"Not enough timesteps ({len(data)}) for seq_len={seq_len} + pred_len={pred_len}"
        )
    inputs = np.stack([data[i:i + seq_len] for i in range(n_windows)])
    targets = np.stack([data[i + seq_len:i + seq_len + pred_len] for i in range(n_windows)])
    return inputs.astype(np.float32), targets.astype(np.float32)


class ERA5WindowDataset(torch.utils.data.Dataset):
    """(input, target) windows for one variable, over one chronological split.

    Pass the train split's fitted `scaler` when building the val/test
    datasets so all splits are normalized consistently.
    """

    def __init__(self, split_name: str, variable: str, seq_len: int, pred_len: int,
                 spatial_stride: int = 1, scaler: Scaler = None):
        files = list_split_files(split_name)
        if not files:
            raise FileNotFoundError(
                f"No processed files found for split '{split_name}'. "
                "Run src/data_prep/preprocess.py first."
            )
        data, self.lat, self.lon = load_variable_grid(files, variable, spatial_stride)
        self.scaler = scaler or Scaler.fit(data)
        data = self.scaler.transform(data)
        self.inputs, self.targets = make_windows(data, seq_len, pred_len)
        self.n_nodes = data.shape[1]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.from_numpy(self.inputs[idx]), torch.from_numpy(self.targets[idx])


def list_year_files(years, processed_dir=PROCESSED_DIR):
    """Processed NetCDF files for explicit years, e.g. for a small CPU demo
    that only needs a year or two, rather than a full train/val/test split.
    """
    files = []
    for year in years:
        year_dir = processed_dir / str(year)
        if year_dir.exists():
            files.extend(sorted(year_dir.glob("*.nc")))
    return files


class WindowTensorDataset(torch.utils.data.Dataset):
    """(input, target) windows already computed in memory (e.g. one
    chronological slice of chronological_demo_datasets' output).

    Exposes `.scaler` like ERA5WindowDataset does, so evaluation code can
    treat both dataset types the same way.
    """

    def __init__(self, inputs: np.ndarray, targets: np.ndarray, scaler: Scaler):
        self.inputs = inputs
        self.targets = targets
        self.scaler = scaler

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.from_numpy(self.inputs[idx]), torch.from_numpy(self.targets[idx])


def chronological_demo_datasets(years, variable: str, seq_len: int, pred_len: int,
                                 spatial_stride: int, train_frac: float = 0.7,
                                 val_frac: float = 0.15):
    """Load a small subset of processed data and split it sequentially (no
    shuffling) into train/val/test windows, for a quick CPU smoke test.

    Unlike ERA5WindowDataset (which uses the real year-based split from
    split.py), this splits one short, contiguous stretch of data by
    position - train is the earliest chunk, then val, then test - so it
    still demonstrates a proper chronological split at a much smaller scale.
    """
    files = list_year_files(years)
    if not files:
        raise FileNotFoundError(
            f"No processed files found for years {years}. "
            "Run src/data_prep/preprocess.py first."
        )

    data, lat, lon = load_variable_grid(files, variable, spatial_stride)
    scaler = Scaler.fit(data)
    data = scaler.transform(data)
    inputs, targets = make_windows(data, seq_len, pred_len)

    n_windows = len(inputs)
    n_train = int(n_windows * train_frac)
    n_val = int(n_windows * val_frac)

    train_ds = WindowTensorDataset(inputs[:n_train], targets[:n_train], scaler)
    val_ds = WindowTensorDataset(inputs[n_train:n_train + n_val], targets[n_train:n_train + n_val], scaler)
    test_ds = WindowTensorDataset(inputs[n_train + n_val:], targets[n_train + n_val:], scaler)

    n_nodes = data.shape[1]
    return train_ds, val_ds, test_ds, n_nodes, scaler


def build_train_and_val(config):
    """(train_ds, val_ds, n_nodes) for either the demo split or the real split.

    Demo mode is selected by the presence of a "demo_years" key in config.
    """
    if "demo_years" in config:
        train_ds, val_ds, _test_ds, n_nodes, _scaler = chronological_demo_datasets(
            config["demo_years"], config["variable"], config["seq_len"], config["pred_len"],
            config["spatial_stride"], config.get("train_frac", 0.7), config.get("val_frac", 0.15),
        )
        return train_ds, val_ds, n_nodes

    train_ds = ERA5WindowDataset(
        "train", config["variable"], config["seq_len"], config["pred_len"], config["spatial_stride"]
    )
    val_ds = ERA5WindowDataset(
        "val", config["variable"], config["seq_len"], config["pred_len"], config["spatial_stride"],
        scaler=train_ds.scaler,
    )
    return train_ds, val_ds, train_ds.n_nodes


def build_train_and_test(config):
    """(train_ds, test_ds, n_nodes) for either the demo split or the real split.

    train_ds is rebuilt only to obtain the same fitted scaler used during
    training (mirrors how evaluation already worked for the real split).
    """
    if "demo_years" in config:
        train_ds, _val_ds, test_ds, n_nodes, _scaler = chronological_demo_datasets(
            config["demo_years"], config["variable"], config["seq_len"], config["pred_len"],
            config["spatial_stride"], config.get("train_frac", 0.7), config.get("val_frac", 0.15),
        )
        return train_ds, test_ds, n_nodes

    train_ds = ERA5WindowDataset(
        "train", config["variable"], config["seq_len"], config["pred_len"], config["spatial_stride"]
    )
    test_ds = ERA5WindowDataset(
        "test", config["variable"], config["seq_len"], config["pred_len"], config["spatial_stride"],
        scaler=train_ds.scaler,
    )
    return train_ds, test_ds, train_ds.n_nodes

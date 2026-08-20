"""Train STLGRU (official architecture - unmodified) on one SAWS weather
variable, reusing the exact same SAWS data pipeline, per-station
normalization, leakage protections, masked-MSE objective, early stopping,
device selection, checkpointing convention, and phase-1 metrics already
used for LiteSTGNN (src/training/train_litestgnn.py) - those helpers are
imported directly below, not duplicated.

STLGRU expects a different input/output tensor convention than LiteSTGNN
(a trailing feature-channel dim: [B, T, N, 1] in, [B, L, N, 1] out) and a
fixed external adjacency matrix (unlike LiteSTGNN, it does not learn its
own graph - see src/data_prep/saws_dataset.py's build_distance_adjacency).
STLGRUAdapter below is a thin shape wrapper that absorbs both differences
so every reused helper function can be called exactly as it already is.

    python -m src.training.train_stlgru_saws --config <path-to-config.json> [--seed N]

Data policy: identical to train_litestgnn.py - see
data/saws_stgnn_dataset/data_leakage_prevention.md. The test split is
built (to prove windows never cross split boundaries) but never scored
here - model selection uses validation masked MSE only.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_prep.config import PROJECT_ROOT
from src.data_prep.saws_dataset import build_saws_splits, build_distance_adjacency
from src.models.STLGRU.model import model as STLGRUModel
from src.training.train_litestgnn import (
    load_config,
    run_dir_for,
    select_device,
    get_git_commit,
    count_trainable_params,
    masked_mse_loss,
    compute_masked_mse_over_loader,
    run_inference,
    compute_phase1_metrics,
    save_predictions,
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "stlgru_saws_baseline.json"
RUNS_DIR = PROJECT_ROOT / "runs"


class STLGRUAdapter(nn.Module):
    """Shape adapter only - wraps the untouched official STLGRU `model`
    class so it exposes forward(x)->[B,L,N] like LiteSTGNN, letting every
    dataset/training/metrics helper from train_litestgnn.py be reused
    verbatim. Never modifies STLGRU's own computation; the tensor STLGRU
    itself receives is exactly [B, T, N, 1] as its official code expects.
    """

    def __init__(self, stlgru_model):
        super().__init__()
        self.stlgru = stlgru_model

    def forward(self, x):
        x = x.unsqueeze(-1)   # [B, T, N] -> [B, T, N, 1] (STLGRU's input_dim=1 channel)
        y = self.stlgru(x)    # [B, L, N, 1]
        return y.squeeze(-1)  # [B, L, N] - matches LiteSTGNN's target convention


def build_model_args(config, n_nodes):
    return SimpleNamespace(
        input_dim=config.get("input_dim", 1),
        n_hid=config.get("n_hid", 64),
        num_nodes=n_nodes,
        out_length=config["pred_len"],
        dropout=config.get("dropout", 0.3),
    )


def build_model(config, n_nodes, adjacency, device):
    args = build_model_args(config, n_nodes)
    adjacency_t = torch.as_tensor(adjacency, dtype=torch.float32, device=device)
    return STLGRUAdapter(STLGRUModel(args, adjacency_t))


def train(config_path=DEFAULT_CONFIG_PATH, seed_override=None, run_dir_override=None):
    config = load_config(config_path)
    seed = seed_override if seed_override is not None else config.get("seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = select_device()
    print(f"[train_stlgru_saws] device: {device}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start_time_iso = datetime.now(timezone.utc).isoformat()
    t_start = time.time()

    train_ds, val_ds, test_ds, n_nodes, scaler, station_order = build_saws_splits(config)
    print(
        f"[train_stlgru_saws] variable={config['variable']} n_nodes={n_nodes} seed={seed} "
        f"train_samples={len(train_ds)} val_samples={len(val_ds)} test_samples={len(test_ds)} "
        "(test built to prove no split-crossing windows, NOT evaluated in this run)"
    )

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

    adjacency = build_distance_adjacency(station_order, epsilon=config.get("adjacency_epsilon", 0.1))
    model = build_model(config, n_nodes, adjacency, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    n_params = count_trainable_params(model)
    print(f"[train_stlgru_saws] trainable parameters: {n_params}")

    max_train_batches = config.get("max_train_batches")
    max_val_batches = config.get("max_val_batches")
    max_epochs = config.get("max_epochs", config.get("epochs", 1))
    early_stopping_patience = config.get("early_stopping_patience")

    run_dir = run_dir_override if run_dir_override is not None else run_dir_for(config_path, seed)
    for sub in ("checkpoints", "metrics", "predictions", "artifacts"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = run_dir / "checkpoints" / "stlgru_best.pt"

    history = []
    best_val_mse, best_epoch, patience_counter = float("inf"), None, 0
    example_input_shape = example_target_shape = None
    total_train_time = 0.0

    for epoch in range(max_epochs):
        epoch_t0 = time.time()
        model.train()
        epoch_losses = []
        for batch_idx, (x, y, y_mask) in enumerate(train_loader):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
            if example_input_shape is None:
                example_input_shape, example_target_shape = tuple(x.shape), tuple(y.shape)
            x, y, y_mask = x.to(device), y.to(device), y_mask.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = masked_mse_loss(pred, y, y_mask)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        train_mse = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        val_mse = compute_masked_mse_over_loader(model, val_loader, device, max_val_batches)
        epoch_time = time.time() - epoch_t0
        total_train_time += epoch_time

        improved = val_mse < best_val_mse
        if improved:
            best_val_mse, best_epoch, patience_counter = val_mse, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "n_nodes": n_nodes,
                "station_order": station_order,
                # stored as a tensor, not a raw numpy array, so this checkpoint
                # still loads under PyTorch 2.6+'s default torch.load(weights_only=True)
                "adjacency": torch.as_tensor(adjacency, dtype=torch.float32),
                "epoch": epoch,
                "val_masked_mse_normalized": val_mse,
            }, best_checkpoint_path)
        else:
            patience_counter += 1

        history.append({
            "epoch": epoch,
            "train_masked_mse_normalized": train_mse,
            "val_masked_mse_normalized": val_mse,
            "n_train_batches": len(epoch_losses),
            "epoch_time_sec": epoch_time,
            "improved": improved,
            "patience_counter": patience_counter,
        })
        print(
            f"[train_stlgru_saws] epoch {epoch}: train_mse={train_mse:.6f} val_mse={val_mse:.6f} "
            f"({epoch_time:.2f}s){' *best*' if improved else ''}"
        )

        if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
            print(f"[train_stlgru_saws] early stopping at epoch {epoch} (patience={early_stopping_patience})")
            break

    if best_epoch is None:
        raise RuntimeError("training produced no epochs - check max_epochs/data")

    print(
        f"[train_stlgru_saws] total training time: {total_train_time:.2f}s over {len(history)} epochs; "
        f"best epoch={best_epoch} (val_mse={best_val_mse:.6f})"
    )

    # Reload the BEST checkpoint (not necessarily the last epoch's weights)
    # into the working model, and independently verify the checkpoint
    # round-trips into a brand-new model instance too.
    best_ckpt = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()
    reloaded = build_model(config, n_nodes, adjacency, device).to(device)
    reloaded.load_state_dict(torch.load(best_checkpoint_path, map_location=device)["model_state_dict"])
    reloaded.eval()
    print(f"[train_stlgru_saws] best checkpoint reloaded and verified: {best_checkpoint_path}")

    t_infer0 = time.time()
    pred_real, true_real, mask_np = run_inference(model, val_loader, scaler, device, max_val_batches)
    inference_time = time.time() - t_infer0
    print(
        f"[train_stlgru_saws] validation inference shapes -> pred={pred_real.shape} true={true_real.shape} "
        f"mask={mask_np.shape} ({inference_time:.2f}s)"
    )

    metrics_payload = compute_phase1_metrics(true_real, pred_real, mask_np, scaler, config)
    metrics_path = run_dir / "metrics" / "validation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[train_stlgru_saws] validation metrics saved ({metrics_payload['result_type']}): {metrics_path}")

    pred_sample_path, pred_full_path = save_predictions(run_dir, station_order, true_real, pred_real, mask_np)
    print(f"[train_stlgru_saws] full validation predictions saved: {pred_full_path}")
    print(f"[train_stlgru_saws] validation prediction sample saved: {pred_sample_path}")

    adjacency_path = run_dir / "artifacts" / "stlgru_adjacency.npy"
    np.save(adjacency_path, adjacency)
    adjacency_labels_path = run_dir / "artifacts" / "stlgru_adjacency_station_order.json"
    with open(adjacency_labels_path, "w") as f:
        json.dump({"station_order": station_order}, f, indent=2)
    print(f"[train_stlgru_saws] fixed distance adjacency saved: {adjacency_path} (row/col order: {adjacency_labels_path})")

    norm_stats_path = run_dir / "artifacts" / "normalization_stats.json"
    with open(norm_stats_path, "w") as f:
        json.dump({"variable": config["variable"], "per_station": scaler.as_dict(station_order)}, f, indent=2)
    print(f"[train_stlgru_saws] per-station normalization stats saved: {norm_stats_path}")

    peak_cuda_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else None

    end_time_iso = datetime.now(timezone.utc).isoformat()
    total_wall_time = time.time() - t_start

    run_metadata = {
        "model": "STLGRU",
        "config_path": str(config_path),
        "config": config,
        "seed": seed,
        "device": str(device),
        "git_commit": get_git_commit(),
        "start_time_utc": start_time_iso,
        "end_time_utc": end_time_iso,
        "total_wall_time_sec": total_wall_time,
        "variable": config["variable"],
        "n_nodes": n_nodes,
        "station_order": station_order,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "test_evaluated": False,
        "example_input_shape": example_input_shape,
        "example_target_shape": example_target_shape,
        "trainable_param_count": n_params,
        "history": history,
        "n_epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_masked_mse_normalized": best_val_mse,
        "early_stopping_patience": early_stopping_patience,
        "checkpoint_selection_metric": "validation masked MSE, normalized/scaled space (same scale/definition as LiteSTGNN's)",
        "total_train_time_sec": total_train_time,
        "mean_epoch_time_sec": total_train_time / len(history) if history else None,
        "validation_inference_time_sec": inference_time,
        "peak_cuda_memory_mb": peak_cuda_mem_mb,
        "checkpoint_path": str(best_checkpoint_path),
        "checkpoint_reload_verified": True,
        "adjacency_path": str(adjacency_path),
        "normalization_stats_path": str(norm_stats_path),
    }
    metadata_path = run_dir / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(run_metadata, f, indent=2)
    print(f"[train_stlgru_saws] run metadata saved: {metadata_path}")

    return {
        "run_dir": run_dir,
        "device": device,
        "checkpoint_path": best_checkpoint_path,
        "metrics_path": metrics_path,
        "pred_sample_path": pred_sample_path,
        "pred_full_path": pred_full_path,
        "metadata_path": metadata_path,
        "adjacency_path": adjacency_path,
        "norm_stats_path": norm_stats_path,
        "metrics_payload": metrics_payload,
        "run_metadata": run_metadata,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override config['seed'] - lets the same config be repeated across independent seeds later without editing the file.",
    )
    args = parser.parse_args()
    train(args.config, seed_override=args.seed)

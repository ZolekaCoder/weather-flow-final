"""Train LiteSTGNN on one SAWS weather variable (temperature, for now).

Same entry point runs unchanged for the local smoke test, the local
real-shape sanity check, and the real A100/Colab baseline run - only the
config file (and available hardware) differ. No Colab-specific code path
exists; PROJECT_ROOT is derived from this file's own location, not any
machine-specific absolute path.

    python -m src.training.train_litestgnn --config <path-to-config.json> [--seed N]

Data policy (see data/saws_stgnn_dataset/data_leakage_prevention.md):
  - Model inputs come from the IDW-imputed SAWS array.
  - Normalization is per-station (src/data_prep/saws_dataset.py's
    PerStationScaler), fit on train-split, originally-observed values only.
  - Training loss and all reported accuracy metrics are masked to exclude
    target cells where the ORIGINAL observation was missing (mask=False).
  - The test split is loaded (to prove windows never cross split
    boundaries) but never scored here - model selection uses validation
    masked MSE only.
"""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data_prep.config import PROJECT_ROOT
from src.data_prep.saws_dataset import build_saws_splits
from src.evaluation import metrics as M
from src.models.LiteSTGNN.model import LiteSTGNN

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "litestgnn_saws_smoke.json"
RUNS_DIR = PROJECT_ROOT / "runs"


def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)


def run_dir_for(config_path, seed):
    """Seed-qualified run directory, so the same config can later be
    repeated across independent seeds without one run overwriting another.
    """
    return RUNS_DIR / Path(config_path).stem / f"seed_{seed}"


def select_device():
    """CUDA if available, else MPS, else CPU - no CUDA-only assumptions.
    Identical on a local Mac and a Colab A100; only the branch taken differs.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_git_commit():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(config, n_nodes):
    return LiteSTGNN(
        n_nodes=n_nodes,
        seq_len=config["seq_len"],
        pred_len=config["pred_len"],
        adj_rank=config.get("adj_rank", 16),
        adj_topk=config.get("adj_topk", 10),
        adj_tau=config.get("adj_tau", 1.2),
        self_loop_alpha=config.get("self_loop_alpha", 0.2),
        gate_mode=config.get("gate_mode", "band"),
        prop_orders=config.get("prop_orders", 1),
        use_spatial_module=config.get("use_spatial_module", True),
    )


def masked_mse_loss(pred, target, mask):
    """Training loss: masked MSE in normalized (per-station-scaled) space.
    Same formula as the physical-unit MSE metric in
    src/evaluation/metrics.py, computed here on scaled tensors for
    training stability - matches the MSE objective used in the published
    Lite-STGNN work. Also used (unchanged) as the validation early-stopping
    metric, so both are on the same scale.
    """
    mask_f = mask.float()
    denom = mask_f.sum().clamp(min=1.0)
    return ((pred - target) ** 2 * mask_f).sum() / denom


def compute_masked_mse_over_loader(model, loader, device, max_batches=None):
    """Normalized-space masked MSE over a whole loader, accumulated across
    batches by summed squared-error / summed valid-cell-count (not a naive
    mean of per-batch means, which would misweight a shorter last batch).
    Used for early stopping / best-checkpoint selection.
    """
    model.eval()
    total_sq_error, total_valid = 0.0, 0.0
    with torch.no_grad():
        for batch_idx, (x, y, y_mask) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x, y, y_mask = x.to(device), y.to(device), y_mask.to(device)
            pred = model(x)
            mask_f = y_mask.float()
            total_sq_error += float(((pred - y) ** 2 * mask_f).sum().item())
            total_valid += float(mask_f.sum().item())
    return total_sq_error / total_valid if total_valid > 0 else float("nan")


def run_inference(model, loader, scaler, device, max_batches=None):
    """Forward pass over a loader, inverse-transformed back to physical
    units via the per-station scaler. Returns (pred_real, true_real,
    observed_mask), all numpy, shape [n_samples, horizon, n_nodes].
    """
    model.eval()
    preds, trues, masks = [], [], []
    with torch.no_grad():
        for batch_idx, (x, y, y_mask) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            pred = model(x.to(device))
            preds.append(pred.cpu().numpy())
            trues.append(y.numpy())
            masks.append(y_mask.numpy())
    pred_norm = np.concatenate(preds, axis=0)
    true_norm = np.concatenate(trues, axis=0)
    mask_np = np.concatenate(masks, axis=0).astype(bool)
    pred_real = scaler.inverse_transform(pred_norm)
    true_real = scaler.inverse_transform(true_norm)
    return pred_real, true_real, mask_np


def compute_phase1_metrics(true_real, pred_real, mask_np, scaler, config):
    """Full Phase 1 metrics: cumulative (1..H) and exact-lead (t+H)
    MAE/MSE/RMSE/MAPE for H in config["eval_horizons"], the full per-lead
    curve, station-normalized NRMSE(1..horizon), and VPT (only computed
    if config["vpt_epsilon"] is set - no default exists, see metrics.py).
    """
    mape_abs_below = config.get("mape_exclude_abs_below", 1.0)
    horizon = pred_real.shape[1]
    eval_horizons = [h for h in config.get("eval_horizons", [6, 12, 24, 48]) if h <= horizon]

    def block(t, p, m):
        return {
            "mae": M.masked_mae(t, p, m),
            "mse": M.masked_mse(t, p, m),
            "rmse": M.masked_rmse(t, p, m),
            "mape_pct": M.masked_mape(t, p, m, eps=mape_abs_below),
        }

    per_lead_full = []
    for h in range(horizon):
        row = block(true_real[:, h], pred_real[:, h], mask_np[:, h])
        row["lead_time_step"] = h + 1
        per_lead_full.append(row)

    cumulative, exact_lead = {}, {}
    for H in eval_horizons:
        cumulative[f"1-{H}h"] = block(true_real[:, :H], pred_real[:, :H], mask_np[:, :H])
        idx = H - 1
        exact_lead[f"t+{H}"] = block(true_real[:, idx : idx + 1], pred_real[:, idx : idx + 1], mask_np[:, idx : idx + 1])

    overall = block(true_real, pred_real, mask_np)
    mape_stats = M.mape_exclusion_stats(true_real, mask_np, abs_below=mape_abs_below)

    nrmse_by_horizon = M.station_normalized_nrmse_per_horizon(true_real, pred_real, mask_np, station_std=scaler.std)

    vpt_epsilon = config.get("vpt_epsilon")
    if vpt_epsilon is None:
        vpt_result = {"vpt_hours": None, "right_censored": None}
        vpt_status = "UNRESOLVED: vpt_epsilon not set in config - see src/evaluation/metrics.py VPT_EPSILON_NOTE."
    else:
        vpt_result = M.valid_prediction_time(nrmse_by_horizon, epsilon=vpt_epsilon)
        vpt_status = f"computed with vpt_epsilon={vpt_epsilon} (explicit project decision, not a placeholder)"

    return {
        "result_type": config.get("run_label", "UNLABELED_RUN"),
        "variable": config["variable"],
        "overall": overall,
        "cumulative": cumulative,
        "exact_lead": exact_lead,
        "per_lead_full": per_lead_full,
        "nrmse_by_horizon": nrmse_by_horizon.tolist(),
        "nrmse_definition_note": M.STATION_NRMSE_DEFINITION_NOTE,
        "vpt_epsilon": vpt_epsilon,
        "vpt": vpt_result,
        "vpt_status": vpt_status,
        "vpt_definition_note": M.VPT_DEFINITION_NOTE,
        "mape_exclusion_stats": mape_stats,
        "mape_note": M.MAPE_EXCLUSION_NOTE,
        "mape_usage_warning": (
            "MAPE is diagnostic/secondary only - never used for training, "
            "early stopping, model selection, or hyperparameter tuning."
        ),
    }


def save_predictions(run_dir, station_order, true_real, pred_real, mask_np, max_samples=3, max_stations=5, filename_prefix="validation"):
    """Full predictions/truth/mask (compressed npz, not truncated) plus a
    small human-readable CSV sample for quick inspection. filename_prefix
    defaults to "validation" so every existing caller (single-run training,
    the tuning runner) keeps producing byte-identical filenames; the final
    test-set evaluator passes filename_prefix="test" so its output can never
    be mistaken for a validation artifact.
    """
    full_path = run_dir / "predictions" / f"{filename_prefix}_predictions_full.npz"
    np.savez_compressed(
        full_path,
        y_true=true_real,
        y_pred=pred_real,
        mask=mask_np,
        station_order=np.array(station_order),
    )

    rows = []
    horizon = pred_real.shape[1]
    for sample in range(min(max_samples, pred_real.shape[0])):
        for h in range(horizon):
            for n in range(min(max_stations, pred_real.shape[2])):
                rows.append({
                    "sample": sample,
                    "lead_time_step": h + 1,
                    "station_index": n,
                    "station_climate_number": station_order[n],
                    "y_true_physical": float(true_real[sample, h, n]),
                    "y_pred_physical": float(pred_real[sample, h, n]),
                    "originally_observed": bool(mask_np[sample, h, n]),
                })
    sample_path = run_dir / "predictions" / f"{filename_prefix}_predictions_sample.csv"
    pd.DataFrame(rows).to_csv(sample_path, index=False)
    return sample_path, full_path


def train(config_path=DEFAULT_CONFIG_PATH, seed_override=None, run_dir_override=None):
    """run_dir_override lets callers (e.g. the tuning runner) place a run's
    outputs at an arbitrary directory instead of the default
    runs/<config_stem>/seed_<seed>/ layout; the single-run CLI below never
    passes it, so its own behavior is unchanged.
    """
    config = load_config(config_path)
    seed = seed_override if seed_override is not None else config.get("seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = select_device()
    print(f"[train_litestgnn] device: {device}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start_time_iso = datetime.now(timezone.utc).isoformat()
    t_start = time.time()

    train_ds, val_ds, test_ds, n_nodes, scaler, station_order = build_saws_splits(config)
    print(
        f"[train_litestgnn] variable={config['variable']} n_nodes={n_nodes} seed={seed} "
        f"train_samples={len(train_ds)} val_samples={len(val_ds)} test_samples={len(test_ds)} "
        "(test built to prove no split-crossing windows, NOT evaluated in this run)"
    )

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

    model = build_model(config, n_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    n_params = count_trainable_params(model)
    print(f"[train_litestgnn] trainable parameters: {n_params}")

    max_train_batches = config.get("max_train_batches")
    max_val_batches = config.get("max_val_batches")
    max_epochs = config.get("max_epochs", config.get("epochs", 1))
    early_stopping_patience = config.get("early_stopping_patience")

    run_dir = run_dir_override if run_dir_override is not None else run_dir_for(config_path, seed)
    for sub in ("checkpoints", "metrics", "predictions", "artifacts"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = run_dir / "checkpoints" / "litestgnn_best.pt"

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
            f"[train_litestgnn] epoch {epoch}: train_mse={train_mse:.6f} val_mse={val_mse:.6f} "
            f"({epoch_time:.2f}s){' *best*' if improved else ''}"
        )

        if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
            print(f"[train_litestgnn] early stopping at epoch {epoch} (patience={early_stopping_patience})")
            break

    if best_epoch is None:
        raise RuntimeError("training produced no epochs - check max_epochs/data")

    print(
        f"[train_litestgnn] total training time: {total_train_time:.2f}s over {len(history)} epochs; "
        f"best epoch={best_epoch} (val_mse={best_val_mse:.6f})"
    )

    # Reload the BEST checkpoint (not necessarily the last epoch's weights)
    # into the working model, and independently verify the checkpoint
    # round-trips into a brand-new model instance too.
    best_ckpt = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()
    reloaded = build_model(config, n_nodes).to(device)
    reloaded.load_state_dict(torch.load(best_checkpoint_path, map_location=device)["model_state_dict"])
    reloaded.eval()
    print(f"[train_litestgnn] best checkpoint reloaded and verified: {best_checkpoint_path}")

    t_infer0 = time.time()
    pred_real, true_real, mask_np = run_inference(model, val_loader, scaler, device, max_val_batches)
    inference_time = time.time() - t_infer0
    print(
        f"[train_litestgnn] validation inference shapes -> pred={pred_real.shape} true={true_real.shape} "
        f"mask={mask_np.shape} ({inference_time:.2f}s)"
    )

    metrics_payload = compute_phase1_metrics(true_real, pred_real, mask_np, scaler, config)
    metrics_path = run_dir / "metrics" / "validation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[train_litestgnn] validation metrics saved ({metrics_payload['result_type']}): {metrics_path}")

    pred_sample_path, pred_full_path = save_predictions(run_dir, station_order, true_real, pred_real, mask_np)
    print(f"[train_litestgnn] full validation predictions saved: {pred_full_path}")
    print(f"[train_litestgnn] validation prediction sample saved: {pred_sample_path}")

    if model.adj is not None:
        with torch.no_grad():
            adjacency = model.adj().cpu().numpy()
        adjacency_path = run_dir / "artifacts" / "learned_adjacency.npy"
        np.save(adjacency_path, adjacency)
        adjacency_labels_path = run_dir / "artifacts" / "learned_adjacency_station_order.json"
        with open(adjacency_labels_path, "w") as f:
            json.dump({"station_order": station_order}, f, indent=2)
        print(f"[train_litestgnn] learned adjacency saved: {adjacency_path} (row/col order: {adjacency_labels_path})")
    else:
        adjacency_path = None
        print("[train_litestgnn] use_spatial_module=False (DLinear-only control) - no adjacency to save")

    norm_stats_path = run_dir / "artifacts" / "normalization_stats.json"
    with open(norm_stats_path, "w") as f:
        json.dump({"variable": config["variable"], "per_station": scaler.as_dict(station_order)}, f, indent=2)
    print(f"[train_litestgnn] per-station normalization stats saved: {norm_stats_path}")

    peak_cuda_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else None

    end_time_iso = datetime.now(timezone.utc).isoformat()
    total_wall_time = time.time() - t_start

    run_metadata = {
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
        "prop_orders": config.get("prop_orders", 1),
        "use_spatial_module": config.get("use_spatial_module", True),
        "history": history,
        "n_epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_masked_mse_normalized": best_val_mse,
        "early_stopping_patience": early_stopping_patience,
        "checkpoint_selection_metric": "validation masked MSE, normalized/scaled space (same scale as the training loss, not the physical-unit metrics)",
        "total_train_time_sec": total_train_time,
        "mean_epoch_time_sec": total_train_time / len(history) if history else None,
        "validation_inference_time_sec": inference_time,
        "peak_cuda_memory_mb": peak_cuda_mem_mb,
        "checkpoint_path": str(best_checkpoint_path),
        "checkpoint_reload_verified": True,
        "learned_adjacency_path": str(adjacency_path) if adjacency_path is not None else None,
        "normalization_stats_path": str(norm_stats_path),
    }
    metadata_path = run_dir / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(run_metadata, f, indent=2)
    print(f"[train_litestgnn] run metadata saved: {metadata_path}")

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

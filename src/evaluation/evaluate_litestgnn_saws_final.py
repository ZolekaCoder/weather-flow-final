#!/usr/bin/env python3
"""Final held-out SAWS test evaluation for the frozen LiteSTGNN candidate.

EVALUATION-ONLY. This script never trains or tunes anything: it loads the
existing best checkpoint for each SAWS variable from

    runs/litestgnn_tuning/<VARIABLE>/K15/seed_42/checkpoints/litestgnn_best.pt

reconstructs the exact model/config that checkpoint was trained with (from
the checkpoint's own embedded "config" - the same resolved config the
tuning runner wrote and trained from), and scores it exactly once against
the untouched test split - the split that was never trained on, never
validated on, and never used for model selection.

Reuses (does not duplicate) existing pipeline code:
  - src.data_prep.saws_dataset.build_saws_splits for data/splits/normalization
    (the PerStationScaler it returns is fit on the train split only, never
    on val/test - see that module's own docstring/policy).
  - src.training.train_litestgnn.build_model to reconstruct the architecture.
  - src.training.train_litestgnn.run_inference for the same physical-unit
    inverse-scaling logic already used for validation.
  - src.training.train_litestgnn.compute_phase1_metrics for the same
    MAE/MSE/RMSE/MAPE/cumulative/exact-lead metric definitions already used
    for validation (MAPE remains diagnostic-only, never used to rank/select
    here - there is no selection step in this script at all).
  - src.training.train_litestgnn.save_predictions for the full-prediction
    npz + CSV sample, with filename_prefix="test" so output is never
    confused with a validation artifact.

Usage:
    python -m src.evaluation.evaluate_litestgnn_saws_final

No arguments: the candidate (K15), seed (42), and variable list (discovered
from the dataset) are frozen for this final evaluation, not configurable
from the command line - this is deliberate, so this script cannot be
repurposed as a generic training/tuning entry point.
"""

import csv
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data_prep.config import PROJECT_ROOT
from src.data_prep.saws_dataset import (
    SAWS_DIR,
    SPLIT_TIMESTAMP_FILES,
    build_saws_splits,
    list_saws_variables,
)
from src.training.train_litestgnn import (
    build_model,
    compute_phase1_metrics,
    count_trainable_params,
    get_git_commit,
    run_inference,
    save_predictions,
    select_device,
)

CANDIDATE_ID = "K15"
SEED = 42
TUNING_ROOT = PROJECT_ROOT / "runs" / "litestgnn_tuning"
FINAL_ROOT = PROJECT_ROOT / "runs" / "litestgnn_final" / f"{CANDIDATE_ID}_seed{SEED}"
CONSOLIDATED_CSV_PATH = FINAL_ROOT / "final_test_results.csv"

# Sanity/testing-cap keys that must never be present in a config used for a
# final reported result (see src/training/tune_litestgnn.py's CAP_KEYS) -
# their presence would mean this checkpoint came from a capped sanity run,
# not a real training run.
FORBIDDEN_CAP_KEYS = ("max_timesteps_per_split", "max_train_batches", "max_val_batches")


class EvaluationOnlyViolation(RuntimeError):
    pass


def _assert_evaluation_only_guard():
    """The real, load-bearing guard: gradient tracking is disabled for the
    *entire process* before any model or data is touched. With autograd off,
    no loss computed anywhere in this process can produce a graph, so
    `.backward()` on it would raise immediately - training cannot silently
    happen even if some future edit accidentally tried.

    (Checking `"torch.optim" in sys.modules` is NOT used here: modern
    PyTorch eagerly imports the `torch.optim` submodule as a side effect of
    plain `import torch`, so that check is always true and would be a
    meaningless, always-firing guard - verified empirically, not assumed.
    This file also never constructs a torch.optim.* optimizer or calls
    .backward() anywhere, by inspection.)
    """
    torch.set_grad_enabled(False)
    if torch.is_grad_enabled():
        raise EvaluationOnlyViolation("failed to globally disable autograd - refusing to proceed.")


def _assert_split_integrity():
    """Independent re-check (not a duplicate of make_windows/window logic)
    that the three chronological split timestamp files never overlap, using
    the exact same files src.data_prep.saws_dataset.split_row_selector reads.
    """
    split_sets = {}
    for split_name, fname in SPLIT_TIMESTAMP_FILES.items():
        ts = pd.read_csv(SAWS_DIR / fname, parse_dates=["timestamp"])
        split_sets[split_name] = set(ts["timestamp"])

    overlaps = {}
    names = list(split_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            shared = split_sets[a] & split_sets[b]
            if shared:
                overlaps[f"{a}∩{b}"] = len(shared)
    if overlaps:
        raise EvaluationOnlyViolation(f"Split timestamp overlap detected - refusing to evaluate: {overlaps}")
    print("[evaluate_litestgnn_saws_final] split integrity check passed: train/val/test timestamps are pairwise disjoint")


def _checkpoint_path(variable):
    return TUNING_ROOT / variable / CANDIDATE_ID / f"seed_{SEED}" / "checkpoints" / "litestgnn_best.pt"


def _assert_all_checkpoints_present(variables):
    missing = [v for v in variables if not _checkpoint_path(v).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {CANDIDATE_ID} seed_{SEED} checkpoint(s) for: {missing}. "
            "This script never trains - run the cross_variable tuning stage for "
            "these variables first (see src/training/tune_litestgnn.py --stage cross_variable), "
            "then re-run this evaluator."
        )


def evaluate_variable(variable, device):
    checkpoint_path = _checkpoint_path(variable)
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = dict(ckpt["config"])

    if config.get("variable") != variable:
        raise EvaluationOnlyViolation(f"{variable}: checkpoint config variable={config.get('variable')!r} mismatch")
    if config.get("seed", SEED) != SEED:
        raise EvaluationOnlyViolation(f"{variable}: checkpoint config seed={config.get('seed')} != {SEED}")
    for cap_key in FORBIDDEN_CAP_KEYS:
        if config.get(cap_key) is not None:
            raise EvaluationOnlyViolation(
                f"{variable}: checkpoint config has {cap_key}={config[cap_key]!r} set - this looks like a "
                "capped/sanity run, not a real training run. Refusing to report it as a final result."
            )

    # Reuse the existing pipeline exactly: same split/window/normalization
    # code as training, driven by the checkpoint's own resolved config.
    train_ds, val_ds, test_ds, n_nodes, scaler, station_order = build_saws_splits(config)
    if n_nodes != ckpt["n_nodes"] or station_order != ckpt["station_order"]:
        raise EvaluationOnlyViolation(f"{variable}: reconstructed station order/count does not match checkpoint")

    model = build_model(config, n_nodes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    if model.training:
        raise EvaluationOnlyViolation(f"{variable}: model did not enter eval() mode")

    test_loader = DataLoader(test_ds, batch_size=config.get("batch_size", 64), shuffle=False)

    t0 = time.time()
    pred_real, true_real, mask_np = run_inference(model, test_loader, scaler, device)
    inference_time_sec = time.time() - t0

    expected_shape = (len(test_ds), config["pred_len"], n_nodes)
    for name, arr in (("pred_real", pred_real), ("true_real", true_real), ("mask_np", mask_np)):
        if arr.shape != expected_shape:
            raise EvaluationOnlyViolation(f"{variable}: {name}.shape={arr.shape} != expected {expected_shape}")

    eval_config = dict(config)
    eval_config["run_label"] = f"FINAL_TEST_{variable}_{CANDIDATE_ID}_seed{SEED}"
    metrics_payload = compute_phase1_metrics(true_real, pred_real, mask_np, scaler, eval_config)
    metrics_payload["candidate_id"] = CANDIDATE_ID
    metrics_payload["seed"] = SEED
    metrics_payload["checkpoint_path"] = str(checkpoint_path)
    metrics_payload["checkpoint_best_epoch"] = ckpt.get("epoch")
    metrics_payload["checkpoint_val_masked_mse_normalized"] = ckpt.get("val_masked_mse_normalized")
    metrics_payload["n_test_samples"] = len(test_ds)
    metrics_payload["inference_time_sec"] = inference_time_sec
    metrics_payload["trainable_param_count"] = count_trainable_params(model)
    metrics_payload["git_commit"] = get_git_commit()
    metrics_payload["split_evaluated"] = "test"
    metrics_payload["test_never_used_for_training_or_selection"] = True

    run_dir = FINAL_ROOT / variable
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics" / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    sample_path, full_path = save_predictions(
        run_dir, station_order, true_real, pred_real, mask_np, filename_prefix="test"
    )

    return {
        "variable": variable,
        "metrics_payload": metrics_payload,
        "n_test_samples": len(test_ds),
        "inference_time_sec": inference_time_sec,
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "full_predictions_path": str(full_path),
        "sample_predictions_path": str(sample_path),
    }


# ---------------------------------------------------------------------------
# Consolidated CSV + final table
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "variable", "candidate_id", "seed", "n_test_samples",
    "overall_mae", "overall_mse", "overall_rmse", "overall_mape_pct",
    "cum_1_6h_mae", "cum_1_6h_mse", "cum_1_6h_rmse", "cum_1_6h_mape_pct",
    "cum_1_12h_mae", "cum_1_12h_mse", "cum_1_12h_rmse", "cum_1_12h_mape_pct",
    "cum_1_24h_mae", "cum_1_24h_mse", "cum_1_24h_rmse", "cum_1_24h_mape_pct",
    "cum_1_48h_mae", "cum_1_48h_mse", "cum_1_48h_rmse", "cum_1_48h_mape_pct",
    "exact_t6_mae", "exact_t6_mse", "exact_t6_rmse", "exact_t6_mape_pct",
    "exact_t12_mae", "exact_t12_mse", "exact_t12_rmse", "exact_t12_mape_pct",
    "exact_t24_mae", "exact_t24_mse", "exact_t24_rmse", "exact_t24_mape_pct",
    "exact_t48_mae", "exact_t48_mse", "exact_t48_rmse", "exact_t48_mape_pct",
    "inference_time_sec", "checkpoint_path", "metrics_json_path", "full_predictions_path",
]


def _block(metrics_payload, section, key=None):
    if section == "overall":
        return metrics_payload.get("overall", {})
    if section == "cumulative":
        return metrics_payload.get("cumulative", {}).get(key, {})
    if section == "exact":
        return metrics_payload.get("exact_lead", {}).get(key, {})
    raise ValueError(section)


def build_csv_row(result):
    m = result["metrics_payload"]
    overall = _block(m, "overall")
    row = {
        "variable": result["variable"],
        "candidate_id": CANDIDATE_ID,
        "seed": SEED,
        "n_test_samples": result["n_test_samples"],
        "overall_mae": overall.get("mae"),
        "overall_mse": overall.get("mse"),
        "overall_rmse": overall.get("rmse"),
        "overall_mape_pct": overall.get("mape_pct"),
        "inference_time_sec": result["inference_time_sec"],
        "checkpoint_path": result["checkpoint_path"],
        "metrics_json_path": result["metrics_path"],
        "full_predictions_path": result["full_predictions_path"],
    }
    for label, key in (("1_6h", "1-6h"), ("1_12h", "1-12h"), ("1_24h", "1-24h"), ("1_48h", "1-48h")):
        b = _block(m, "cumulative", key)
        row[f"cum_{label}_mae"] = b.get("mae")
        row[f"cum_{label}_mse"] = b.get("mse")
        row[f"cum_{label}_rmse"] = b.get("rmse")
        row[f"cum_{label}_mape_pct"] = b.get("mape_pct")
    for label, key in (("t6", "t+6"), ("t12", "t+12"), ("t24", "t+24"), ("t48", "t+48")):
        b = _block(m, "exact", key)
        row[f"exact_{label}_mae"] = b.get("mae")
        row[f"exact_{label}_mse"] = b.get("mse")
        row[f"exact_{label}_rmse"] = b.get("rmse")
        row[f"exact_{label}_mape_pct"] = b.get("mape_pct")
    return row


def write_consolidated_csv(results):
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    with open(CONSOLIDATED_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(build_csv_row(result))
    print(f"[evaluate_litestgnn_saws_final] consolidated CSV written: {CONSOLIDATED_CSV_PATH}")


def print_final_table(results):
    print("\n[evaluate_litestgnn_saws_final] FINAL HELD-OUT TEST RESULTS "
          f"(candidate={CANDIDATE_ID}, seed={SEED}) - MAPE is diagnostic only:")
    header = (f"{'variable':<16}{'n_test':<8}{'MAE':<10}{'MSE':<10}{'RMSE':<10}"
              f"{'t+48 MAE':<10}{'MAPE%':<9}{'infer_s':<9}")
    print(header)
    print("-" * len(header))
    for result in results:
        m = result["metrics_payload"]
        overall = _block(m, "overall")
        t48 = _block(m, "exact", "t+48")
        print(
            f"{result['variable']:<16}{result['n_test_samples']:<8}"
            f"{overall.get('mae', float('nan')):<10.4f}{overall.get('mse', float('nan')):<10.4f}"
            f"{overall.get('rmse', float('nan')):<10.4f}{t48.get('mae', float('nan')):<10.4f}"
            f"{overall.get('mape_pct', float('nan')):<9.2f}{result['inference_time_sec']:<9.2f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _assert_evaluation_only_guard()
    _assert_split_integrity()

    variables = list_saws_variables()
    print(f"[evaluate_litestgnn_saws_final] discovered SAWS variables: {variables}")
    _assert_all_checkpoints_present(variables)
    print(f"[evaluate_litestgnn_saws_final] all {len(variables)} {CANDIDATE_ID} seed_{SEED} checkpoints found")

    device = select_device()
    print(f"[evaluate_litestgnn_saws_final] device: {device}")

    results = []
    for variable in variables:
        print(f"[evaluate_litestgnn_saws_final] evaluating {variable} on the held-out test split ...")
        result = evaluate_variable(variable, device)
        results.append(result)
        overall = _block(result["metrics_payload"], "overall")
        print(
            f"[evaluate_litestgnn_saws_final] {variable}: n_test={result['n_test_samples']} "
            f"overall_MAE={overall.get('mae'):.4f} overall_MSE={overall.get('mse'):.4f} "
            f"inference_time={result['inference_time_sec']:.2f}s"
        )

    if len(results) != len(variables) or len(results) != 6:
        raise EvaluationOnlyViolation(f"expected 6 completed variables, got {len(results)}")

    write_consolidated_csv(results)
    print_final_table(results)
    print(f"\n[evaluate_litestgnn_saws_final] all {len(results)} variables completed successfully - "
          "test split evaluated exactly once each, never used for training or model selection.")


if __name__ == "__main__":
    main()

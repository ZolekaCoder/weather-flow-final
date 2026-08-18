"""Hyperparameter-tuning orchestration for LiteSTGNN SAWS experiments.

This module reuses src/training/train_litestgnn.py's train() for every
actual training run - it does not reimplement any training, data-loading,
normalization, masking, checkpointing, or metrics logic. Its only job is
to resolve per-experiment configs, place each run's outputs in its own
directory, skip/resume already-completed runs, and maintain a running
tuning_summary.csv - so a Colab disconnect can be recovered from by simply
re-running the same command.

Experiment grids (which hyperparameters to vary) live as plain JSON files
under configs/tuning/ (e.g. configs/tuning/screen.json), not in this file,
so Stage 1's factors can be edited/extended without touching Python. The
cross_variable/multiseed/lr stages look up a candidate id's overrides from
configs/tuning/screen.json and cross them with variables/seeds/learning
rates given on the command line - also no Python edits required.

Usage:
    python -m src.training.tune_litestgnn --stage screen
    python -m src.training.tune_litestgnn --stage cross_variable --candidates B0,K5,R8
    python -m src.training.tune_litestgnn --stage multiseed --candidates K5,R8 --seeds 42 123 456
    python -m src.training.tune_litestgnn --stage lr --candidates B0 --lrs 0.0005 0.001 0.002

Validation only: this runner never reads or scores the test split - it
only ever calls train_litestgnn.train(), which builds the test split solely
to prove windows don't cross split boundaries and never evaluates it. As an
explicit extra guard, every run recorded here (freshly trained or loaded
from a prior completed run) has its run_metadata "test_evaluated" field
asserted to be False before it is written into the summary.
"""

import argparse
import copy
import csv
import json
import sys
import traceback
from pathlib import Path

from src.data_prep.config import PROJECT_ROOT
from src.data_prep.saws_dataset import list_saws_variables
from src.training import train_litestgnn

TUNING_ROOT = PROJECT_ROOT / "runs" / "litestgnn_tuning"
GRID_DIR = PROJECT_ROOT / "configs" / "tuning"
SUMMARY_PATH = TUNING_ROOT / "tuning_summary.csv"

SUMMARY_FIELDS = [
    "experiment_id", "variable", "seed",
    "adj_rank", "adj_topk", "adj_tau", "prop_orders", "self_loop_alpha",
    "learning_rate", "batch_size", "use_spatial_module",
    "parameter_count", "best_epoch", "epochs_run", "training_seconds",
    "best_val_masked_mse_normalized",
    "overall_mae", "overall_mse", "overall_rmse",
    "cum_1_6h_mae", "cum_1_12h_mae", "cum_1_24h_mae", "cum_1_48h_mae",
    "exact_t48_mae",
    "run_dir", "status",
]


# ---------------------------------------------------------------------------
# Grid / config resolution
# ---------------------------------------------------------------------------

def load_grid(stage_name):
    path = GRID_DIR / f"{stage_name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No grid definition at {path}. Sweep grids are plain JSON files "
            "under configs/tuning/ - add or edit one there, no Python change needed."
        )
    with open(path) as f:
        return json.load(f)


def lookup_candidate_overrides(screen_grid, candidate_id):
    for exp in screen_grid["experiments"]:
        if exp["id"] == candidate_id:
            return exp.get("overrides", {})
    known = ", ".join(exp["id"] for exp in screen_grid["experiments"])
    raise ValueError(f"Unknown candidate id '{candidate_id}' - not found in configs/tuning/screen.json. Known ids: {known}")


def resolve_config(base_config, overrides, variable=None, seed=None, run_label=None):
    cfg = copy.deepcopy(base_config)
    cfg.update(overrides)
    if variable is not None:
        cfg["variable"] = variable
    if seed is not None:
        cfg["seed"] = seed
    if run_label is not None:
        cfg["run_label"] = run_label
    return cfg


# ---------------------------------------------------------------------------
# Per-experiment execution
# ---------------------------------------------------------------------------

def is_run_complete(run_dir):
    return (run_dir / "run_metadata.json").exists() and (run_dir / "metrics" / "validation_metrics.json").exists()


def _assert_test_not_evaluated(meta, context):
    if meta.get("test_evaluated") is not False:
        raise RuntimeError(
            f"Refusing to record {context}: run_metadata['test_evaluated'] is "
            f"{meta.get('test_evaluated')!r}, not False. Tuning must never select "
            "on test-set metrics."
        )


CAP_KEYS = ("max_timesteps_per_split", "max_epochs", "max_train_batches", "max_val_batches")


def _verify_config_consistency(existing_config, expected_config, keys, context):
    """Compares the override keys AND the sanity/testing cap keys (CAP_KEYS)
    between an existing completed run and what this invocation would produce.
    Checking the caps too means a tiny/capped local sanity run can never be
    silently mistaken for a completed real run (or vice versa) just because
    they share a run directory - a mismatch raises instead of skipping.
    """
    mismatches = {}
    for key in set(keys) | set(CAP_KEYS):
        if existing_config.get(key) != expected_config.get(key):
            mismatches[key] = (expected_config.get(key), existing_config.get(key))
    if mismatches:
        raise RuntimeError(
            f"Existing completed run at {context} was built with different "
            f"config than requested (expected, found): {mismatches}. "
            "Use --force to intentionally overwrite it."
        )


def cum_mae(metrics, key):
    block = metrics.get("cumulative", {}).get(key)
    return block["mae"] if block else None


def exact_mae(metrics, key):
    block = metrics.get("exact_lead", {}).get(key)
    return block["mae"] if block else None


def build_row(meta, metrics, experiment_id, variable, seed, run_dir, status):
    cfg = meta.get("config", {})
    overall = metrics.get("overall", {})
    return {
        "experiment_id": experiment_id,
        "variable": variable,
        "seed": seed,
        "adj_rank": cfg.get("adj_rank"),
        "adj_topk": cfg.get("adj_topk"),
        "adj_tau": cfg.get("adj_tau"),
        "prop_orders": cfg.get("prop_orders"),
        "self_loop_alpha": cfg.get("self_loop_alpha"),
        "learning_rate": cfg.get("learning_rate"),
        "batch_size": cfg.get("batch_size"),
        "use_spatial_module": cfg.get("use_spatial_module", True),
        "parameter_count": meta.get("trainable_param_count"),
        "best_epoch": meta.get("best_epoch"),
        "epochs_run": meta.get("n_epochs_run"),
        "training_seconds": meta.get("total_train_time_sec"),
        "best_val_masked_mse_normalized": meta.get("best_val_masked_mse_normalized"),
        "overall_mae": overall.get("mae"),
        "overall_mse": overall.get("mse"),
        "overall_rmse": overall.get("rmse"),
        "cum_1_6h_mae": cum_mae(metrics, "1-6h"),
        "cum_1_12h_mae": cum_mae(metrics, "1-12h"),
        "cum_1_24h_mae": cum_mae(metrics, "1-24h"),
        "cum_1_48h_mae": cum_mae(metrics, "1-48h"),
        "exact_t48_mae": exact_mae(metrics, "t+48"),
        "run_dir": str(run_dir),
        "status": status,
    }


def failed_row(experiment_id, variable, seed, cfg, run_dir, error):
    return {
        "experiment_id": experiment_id,
        "variable": variable,
        "seed": seed,
        "adj_rank": cfg.get("adj_rank"),
        "adj_topk": cfg.get("adj_topk"),
        "adj_tau": cfg.get("adj_tau"),
        "prop_orders": cfg.get("prop_orders"),
        "self_loop_alpha": cfg.get("self_loop_alpha"),
        "learning_rate": cfg.get("learning_rate"),
        "batch_size": cfg.get("batch_size"),
        "use_spatial_module": cfg.get("use_spatial_module", True),
        "parameter_count": None, "best_epoch": None, "epochs_run": None, "training_seconds": None,
        "best_val_masked_mse_normalized": None,
        "overall_mae": None, "overall_mse": None, "overall_rmse": None,
        "cum_1_6h_mae": None, "cum_1_12h_mae": None, "cum_1_24h_mae": None, "cum_1_48h_mae": None,
        "exact_t48_mae": None,
        "run_dir": str(run_dir),
        "status": f"failed: {error}"[:200],
    }


def load_completed_row(run_dir, experiment_id, variable, seed):
    with open(run_dir / "run_metadata.json") as f:
        meta = json.load(f)
    with open(run_dir / "metrics" / "validation_metrics.json") as f:
        metrics = json.load(f)
    _assert_test_not_evaluated(meta, context=str(run_dir))
    return meta, build_row(meta, metrics, experiment_id, variable, seed, run_dir, status="skipped_existing")


def run_experiment(experiment_id, variable, seed, base_config, overrides, force=False,
                    max_timesteps_per_split=None, max_epochs_override=None,
                    max_train_batches=None, max_val_batches=None):
    run_dir = TUNING_ROOT / variable / experiment_id / f"seed_{seed}"
    cfg = resolve_config(base_config, overrides, variable=variable, seed=seed,
                          run_label=f"TUNING_{experiment_id}_{variable}_seed{seed}")
    # Sanity/testing caps only - never set for a real sweep.
    if max_timesteps_per_split is not None:
        cfg["max_timesteps_per_split"] = max_timesteps_per_split
    if max_epochs_override is not None:
        cfg["max_epochs"] = max_epochs_override
    if max_train_batches is not None:
        cfg["max_train_batches"] = max_train_batches
    if max_val_batches is not None:
        cfg["max_val_batches"] = max_val_batches

    if is_run_complete(run_dir) and not force:
        meta, row = load_completed_row(run_dir, experiment_id, variable, seed)
        _verify_config_consistency(meta.get("config", {}), cfg, overrides.keys(), context=str(run_dir))
        return row, "skipped_existing"

    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = run_dir / "resolved_config.json"
    with open(resolved_config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    result = train_litestgnn.train(config_path=resolved_config_path, seed_override=seed, run_dir_override=run_dir)

    _assert_test_not_evaluated(result["run_metadata"], context=f"{experiment_id}/{variable}/seed_{seed}")
    return build_row(result["run_metadata"], result["metrics_payload"], experiment_id, variable, seed, run_dir, status="completed_new"), "completed_new"


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def load_summary_rows():
    rows = {}
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH, newline="") as f:
            for r in csv.DictReader(f):
                seed = int(r["seed"]) if r.get("seed") not in (None, "") else None
                rows[(r["experiment_id"], r["variable"], seed)] = r
    return rows


def _sort_key(row):
    try:
        return float(row.get("best_val_masked_mse_normalized"))
    except (TypeError, ValueError):
        return float("inf")


def write_summary(rows):
    TUNING_ROOT.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=_sort_key)
    with open(SUMMARY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in ordered:
            writer.writerow({k: row.get(k) for k in SUMMARY_FIELDS})


def print_summary_table(rows):
    ordered = sorted(rows.values(), key=_sort_key)
    print("\n[tune_litestgnn] summary (sorted by normalized validation MSE):")
    header = f"{'id':<10}{'variable':<14}{'seed':<6}{'val_mse_norm':<14}{'status':<20}"
    print(header)
    print("-" * len(header))
    for row in ordered:
        val_mse = row.get("best_val_masked_mse_normalized")
        val_mse_str = f"{float(val_mse):.6f}" if val_mse not in (None, "") else "n/a"
        print(f"{row.get('experiment_id',''):<10}{row.get('variable',''):<14}{str(row.get('seed','')):<6}{val_mse_str:<14}{row.get('status',''):<20}")


# ---------------------------------------------------------------------------
# Stage job-list construction
# ---------------------------------------------------------------------------

def jobs_for_screen(grid, base_config):
    return [
        (exp["id"], variable, seed, exp.get("overrides", {}))
        for exp in grid["experiments"]
        for variable in grid["variables"]
        for seed in grid["seeds"]
    ]


def jobs_for_cross_variable(grid, base_config, candidates, seeds):
    variables = list_saws_variables()
    return [
        (cid, variable, seed, lookup_candidate_overrides(grid, cid))
        for cid in candidates
        for variable in variables
        for seed in seeds
    ]


def jobs_for_multiseed(grid, base_config, candidates, seeds):
    variable = base_config["variable"]
    return [
        (cid, variable, seed, lookup_candidate_overrides(grid, cid))
        for cid in candidates
        for seed in seeds
    ]


def jobs_for_lr(grid, base_config, candidates, lrs):
    variable = base_config["variable"]
    seed = base_config.get("seed", 42)
    jobs = []
    for cid in candidates:
        base_overrides = lookup_candidate_overrides(grid, cid)
        for lr in lrs:
            overrides = dict(base_overrides)
            overrides["learning_rate"] = lr
            jobs.append((f"{cid}_lr{lr}", variable, seed, overrides))
    return jobs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=["screen", "cross_variable", "multiseed", "lr"])
    parser.add_argument("--candidates", default=None, help="Comma-separated experiment ids from configs/tuning/screen.json (required for cross_variable/multiseed/lr)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Seeds (required for --stage multiseed; optional override for cross_variable)")
    parser.add_argument("--lrs", nargs="+", type=float, default=None, help="Learning rates (required for --stage lr)")
    parser.add_argument("--force", action="store_true", help="Retrain even if a completed run already exists at the target directory")
    parser.add_argument("--max-timesteps-per-split", type=int, default=None, help="Sanity/testing cap only - do not use for a real sweep")
    parser.add_argument("--max-epochs", type=int, default=None, help="Sanity/testing cap only - do not use for a real sweep")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Sanity/testing cap only - do not use for a real sweep")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Sanity/testing cap only - do not use for a real sweep")
    args = parser.parse_args(argv)

    if args.stage in ("cross_variable", "multiseed", "lr") and not args.candidates:
        parser.error(f"--stage {args.stage} requires --candidates")
    if args.stage == "multiseed" and not args.seeds:
        parser.error("--stage multiseed requires --seeds")
    if args.stage == "lr" and not args.lrs:
        parser.error("--stage lr requires --lrs")
    return args


def main(argv=None):
    args = parse_args(argv)
    grid = load_grid("screen")
    base_config = train_litestgnn.load_config(PROJECT_ROOT / grid["base_config"])
    candidates = args.candidates.split(",") if args.candidates else None

    if args.stage == "screen":
        jobs = jobs_for_screen(grid, base_config)
    elif args.stage == "cross_variable":
        jobs = jobs_for_cross_variable(grid, base_config, candidates, args.seeds or grid["seeds"])
    elif args.stage == "multiseed":
        jobs = jobs_for_multiseed(grid, base_config, candidates, args.seeds)
    elif args.stage == "lr":
        jobs = jobs_for_lr(grid, base_config, candidates, args.lrs)
    else:
        raise AssertionError(f"unhandled stage {args.stage}")

    rows = load_summary_rows()
    print(f"[tune_litestgnn] stage={args.stage}: {len(jobs)} experiment(s) to check/run")

    for experiment_id, variable, seed, overrides in jobs:
        try:
            row, status = run_experiment(
                experiment_id, variable, seed, base_config, overrides,
                force=args.force,
                max_timesteps_per_split=args.max_timesteps_per_split,
                max_epochs_override=args.max_epochs,
                max_train_batches=args.max_train_batches,
                max_val_batches=args.max_val_batches,
            )
        except Exception as exc:  # noqa: BLE001 - keep the sweep going past one bad config
            traceback.print_exc()
            expected_config = resolve_config(base_config, overrides, variable=variable, seed=seed)
            run_dir = TUNING_ROOT / variable / experiment_id / f"seed_{seed}"
            row = failed_row(experiment_id, variable, seed, expected_config, run_dir, error=str(exc))
            status = "failed"

        rows[(experiment_id, variable, seed)] = row
        write_summary(rows)
        print(f"[tune_litestgnn] {status}: id={experiment_id} variable={variable} seed={seed}")

    print_summary_table(rows)
    print(f"\n[tune_litestgnn] summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main(sys.argv[1:])

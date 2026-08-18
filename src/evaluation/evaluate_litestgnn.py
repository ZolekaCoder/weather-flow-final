"""Evaluate a trained LiteSTGNN checkpoint on the ERA5 test split.

Loads runs/<config name>/checkpoints/litestgnn_best.pt (written by
src/training/train_litestgnn.py for the same --config), runs it on the
"test" split, and writes metrics + a sample of predictions into
runs/<config name>/.
"""

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data_prep.dataset import build_train_and_test
from src.training.train_litestgnn import DEFAULT_CONFIG_PATH, build_model, load_config, run_dir_for


def masked_mape(pred, true, eps=1e-3):
    mask = np.abs(true) > eps
    return float(np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100)


def evaluate(config_path):
    config = load_config(config_path)
    run_dir = run_dir_for(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = run_dir / "checkpoints" / "litestgnn_best.pt"
    print(f"[evaluate] config: {config_path}")
    print(f"[evaluate] checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    train_ds, test_ds, n_nodes = build_train_and_test(config)
    print(f"[evaluate] test samples: {len(test_ds)}")
    print(f"[evaluate] spatial nodes: {n_nodes}")

    max_test_batches = config.get("max_test_batches")
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    model = build_model(config, n_nodes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    abs_errors, sq_errors = [], []
    true_all, pred_all = [], []
    prediction_rows = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            if max_test_batches is not None and batch_idx >= max_test_batches:
                break
            x, y = x.to(device), y.to(device)
            pred = model(x)

            pred_real = train_ds.scaler.inverse_transform(pred.cpu().numpy())
            true_real = train_ds.scaler.inverse_transform(y.cpu().numpy())

            abs_errors.append(np.abs(pred_real - true_real))
            sq_errors.append((pred_real - true_real) ** 2)
            true_all.append(true_real)
            pred_all.append(pred_real)

            if batch_idx == 0:
                for h in range(pred_real.shape[1]):
                    for n in range(min(5, pred_real.shape[2])):
                        prediction_rows.append({
                            "result_type": "REAL_MODEL_OUTPUT",
                            "sample": 0,
                            "horizon_step": h + 1,
                            "node": n,
                            "y_true": float(true_real[0, h, n]),
                            "y_pred": float(pred_real[0, h, n]),
                        })

    abs_errors = np.concatenate(abs_errors)
    sq_errors = np.concatenate(sq_errors)
    true_all = np.concatenate(true_all)
    pred_all = np.concatenate(pred_all)

    metrics = {
        "result_type": "REAL_MODEL_OUTPUT",
        "variable": config["variable"],
        "mae": float(abs_errors.mean()),
        "rmse": float(np.sqrt(sq_errors.mean())),
        "mape_pct": masked_mape(pred_all, true_all),
    }

    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(run_dir / "metrics" / "test_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(run_dir / "predictions" / "test_predictions.csv", index=False)

    print(f"[evaluate] test MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} MAPE={metrics['mape_pct']:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    evaluate(args.config)

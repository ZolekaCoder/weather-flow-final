# configs

Experiment settings (hyperparameters, dataset paths, run options).

- `litestgnn.json` — hyperparameters for `src/training/train_litestgnn.py` (target variable, sequence/horizon length, spatial subsampling, model/optimizer settings).
- `litestgnn_demo.json` — same shape, but a small 2011-only subset, few spatial nodes, 1 epoch, and capped batch counts, for a fast CPU smoke test (`python -m src.training.run_litestgnn_demo`).

# src/evaluation

Metrics and comparison tables for STLGRU vs. baseline.

Reads from `runs/`, writes summary tables to `results/`. No training logic here.

## LiteSTGNN

`evaluate_litestgnn.py` loads `runs/litestgnn/checkpoints/litestgnn_best.pt`, runs it on the "test" split, and writes `runs/litestgnn/metrics/test_metrics.csv` + `runs/litestgnn/predictions/test_predictions.csv`.

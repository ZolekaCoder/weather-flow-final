# src/training

Training loop and checkpointing logic.

Consumes `src/models/` and `src/data_prep/` output; writes checkpoints to `runs/`. No model definitions or metrics here.

## LiteSTGNN on SAWS (temperature)

`train_litestgnn.py` trains LiteSTGNN on one SAWS variable at a time (`src/data_prep/saws_dataset.py`), with a mask-aware loss and Phase 1 metrics (`src/evaluation/metrics.py`). One entry point, three configs, same code path:

- `configs/litestgnn_saws_smoke.json` — tiny/fast, proves the pipeline runs at all.
- `configs/litestgnn_saws_realshape_sanity.json` — real shapes (96h in / 48h out, per-station scaler, rank=16, topk=10, flat gating) on a small truncated/capped slice, for a fast local check before the real run.
- `configs/litestgnn_saws_baseline.json` — the real protocol, full dataset, 100-epoch budget with early stopping. Intended for Colab/A100, not a local full run.

```bash
python -m src.training.train_litestgnn --config configs/litestgnn_saws_realshape_sanity.json
python -m src.training.train_litestgnn --config configs/litestgnn_saws_baseline.json --seed 42
```

Runs land in `runs/<config-name>/seed_<seed>/` (checkpoints, metrics, predictions, artifacts, `run_metadata.json`) — seed-qualified so the same config can later be repeated across independent seeds without collisions.

## Colab / A100

No Colab-specific code exists — same repo, same entry point. Minimal launch procedure:

```bash
# 1. clone (or open an already-cloned copy of) the repo
git clone <this-repo-url> weatherflow && cd weatherflow
# 2. get data/saws_stgnn_dataset/ onto the runtime (small enough to just be
#    part of the repo checkout; otherwise copy it in or mount Drive and symlink it)
# 3. install requirements
pip install -r requirements.txt
# 4. verify the GPU is visible
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 5. run the real baseline with the same entry point used locally
python -m src.training.train_litestgnn --config configs/litestgnn_saws_baseline.json --seed 42
```

`select_device()` in `train_litestgnn.py` already picks CUDA first, then MPS, then CPU — nothing else changes between a local Mac run and this A100 run except which branch it takes and how big the config is.

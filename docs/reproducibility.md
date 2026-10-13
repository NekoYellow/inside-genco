# Reproducibility guide

## Inputs

`assets/manifests/case118-assets.lock.json` pins the primary case118 data and model revision. `assets/manifests/cross-network-assets.lock.json` pins the added seed, model-width, and case30 controls. Every file entry records its remote path, local path, byte size, and content hash.

The frozen operating-condition selections are in `assets/operating-conditions/`. Their dependency hashes bind each selection to its asset manifest, official test split, topology index, and group split.

## Environment

```bash
bash scripts/setup_vendor.sh
bash scripts/setup_env.sh cpu
```

The vendor setup checks out `gridfm-graphkit` at the revision in `vendor/graphkit-source.json`. The environment script creates `.venv`, installs pinned Python dependencies, and records the resolved environment under `outputs/`.

## Download

```bash
.venv/bin/python scripts/download_assets.py \
  --lock assets/manifests/case118-assets.lock.json --scope full
.venv/bin/python scripts/download_assets.py \
  --lock assets/manifests/cross-network-assets.lock.json --scope full
```

## Run

```bash
for variant in \
  case118_base_seed0 case118_base_seed1 \
  case118_small_seed0 case30_base_seed0
do
  .venv/bin/python scripts/run_trajectory_analysis.py --variant "$variant"
done

.venv/bin/python scripts/summarize_trajectory_analysis.py

.venv/bin/python scripts/run_operator_substitution.py \
  --variant case118_base_seed0 --batch-size 115
.venv/bin/python scripts/summarize_operator_substitution.py
.venv/bin/python scripts/report_parameter_sharing.py
```

Each checkpoint run writes:

- `manifest.json`: checkpoint, inputs, hashes, runtime, and frozen-state checks
- `physical_trajectory.csv`: all per-condition physical quality metrics after each correction
- `physical_updates.csv`: decoded voltage and generation movement between adjacent corrections
- `step_summary.csv`: median, mean, and 95th percentile residual by correction
- `physical_metric_summary.csv` and `physical_update_summary.csv`: long-form compact physical summaries
- `cka.csv`: cross-correction and within-correction CKA
- `ablation.csv`: final residual change from one-correction bypasses
- `local_change_vs_importance.csv`: step-level association table
- `summary.json`: compact checkpoint summary and paired bootstrap intervals

The operator-substitution run replaces the graph Transformer and both node-type LayerNorms at target correction `t` with those learned for source correction `s`. Corrections 2–12 are shape-compatible, producing an 11×11 matrix. It writes condition-level physical metrics, a compact matrix summary, a frozen checkpoint hash, and a complete/failed manifest under `outputs/operator-substitution/`.

## Trainable parameter-sharing controls

Two GridFM plugin models are provided without changing vendored upstream code:

- `GNS_heterogeneous_tied_tail`: correction 1 is independent; corrections 2--12 share one graph operator and both LayerNorms.
- `GNS_heterogeneous_shared_graph_step_norms`: the same shared graph operator, with separate per-step LayerNorms.

The matched nine-run protocol preserves the released case118 Base width, correction count, losses, optimizer, and epoch budget. Its fixed-split preparation step, dry-run validation, and long-running commands are documented in [training-controls.md](training-controls.md). Training is intentionally not launched by the analysis workflow. A trained tied/untied accuracy claim requires identical splits, normalizer, update budget, checkpoint-selection metric, precision, and evaluation conditions. `results/parameter-sharing.json` is only architecture-level accounting; it is not a trained comparison.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

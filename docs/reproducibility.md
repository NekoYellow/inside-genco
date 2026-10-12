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
```

Each checkpoint run writes:

- `manifest.json`: checkpoint, inputs, hashes, runtime, and frozen-state checks
- `physical_trajectory.csv`: per-condition residual after each correction
- `step_summary.csv`: median, mean, and 95th percentile residual by correction
- `cka.csv`: cross-correction and within-correction CKA
- `local_updates.csv`: condition-level direct update summaries
- `node_updates.csv`: one relative update per condition, correction, node type, and stable device ID
- `ablations.csv`: final residual change from one-correction bypasses
- `local_change_vs_importance.csv`: step-level association table
- `summary.json`: compact checkpoint summary and paired bootstrap intervals

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

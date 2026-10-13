# Reproduce the frozen-checkpoint analysis

Follow the environment and download commands in [README.md](../README.md). Asset manifests pin upstream revisions, file sizes, and hashes. All experiments use inference on released checkpoints.

```bash
for variant in case118_base_seed0 case118_base_seed1 case118_small_seed0 case30_base_seed0
do
  .venv/bin/python scripts/run_trajectory_analysis.py --variant "$variant"
done
.venv/bin/python scripts/summarize_trajectory_analysis.py
.venv/bin/python scripts/run_operator_substitution.py --variant case118_base_seed0 --batch-size 115
.venv/bin/python scripts/summarize_operator_substitution.py
.venv/bin/python -m unittest discover -s tests -v
```

Each trajectory run writes physical quality and adjacent decoded-variable changes, cross-step/within-step CKA, whole-correction bypass effects, and condition-bootstrap association statistics under `outputs/trajectory-analysis/<variant>/`. Its manifest records checkpoint/input hashes, runtime, completion status, and checkpoint-state immutability checks. Summarization generates compact tables under `results/` and the trajectory, CKA, and bypass figures under `figures/`.

# Reproduce the experiments

## Setup

Follow the environment and download commands in [README.md](../README.md#reproduce).

## Frozen-checkpoint analysis

```bash
for variant in case118_base_seed0 case118_base_seed1 case118_small_seed0 case30_base_seed0
do
  .venv/bin/python scripts/run_trajectory_analysis.py --variant "$variant"
done
.venv/bin/python scripts/summarize_trajectory_analysis.py
.venv/bin/python scripts/run_operator_substitution.py --variant case118_base_seed0 --batch-size 115
for component in graph norm
do
  .venv/bin/python scripts/run_operator_substitution.py --variant case118_base_seed0 --component "$component" --batch-size 64
done
.venv/bin/python scripts/summarize_operator_substitution.py --components
.venv/bin/python -m unittest discover -s tests -v
```

## Training controls: 50 → 100 epochs

```bash
.venv/bin/python scripts/prepare_training_controls.py
for architecture in untied tied_tail shared_graph_step_norms
do
  for seed in 0 1
  do
    .venv/bin/python scripts/run_training_control.py --architecture "$architecture" --seed "$seed"
    .venv/bin/python scripts/run_training_control.py --architecture "$architecture" --seed "$seed" --resume-to-epoch 100
  done
done
```

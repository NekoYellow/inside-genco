# Matched Untied and Recurrent-Tail Training Controls

## Comparison design

The confirmatory experiment contains nine runs: three architectures evaluated with initialization seeds 0, 1, and 2. All runs use the published case118 Base seed-0 train/validation/test split, the corresponding frozen normalizer, 12 correction calls, hidden size 48, the same losses and weights, AdamW settings, batch size, 200 training epochs, and the same validation checkpoint-selection metric.

| Architecture | Model type | Unique parameters |
| --- | --- | ---: |
| Untied | `GNS_heterogeneous` | 20,069,187 |
| Strict recurrent tail | `GNS_heterogeneous_tied_tail` | 2,094,147 |
| Shared graph with step-specific norms | `GNS_heterogeneous_shared_graph_step_norms` | 2,109,507 |

Correction 1 remains an independent width-changing stem in both recurrent controls. Corrections 2–12 share the graph operator; the strict control also shares bus/generator LayerNorms, whereas the diagnostic control retains separate LayerNorms at each depth.

For a given seed, all three models first follow the same module-construction and random-initialization sequence. The tied variants then alias corrections 2–12 to the initialized correction-2 modules. This aligns the stem, shared heads, and recurrent operator initialization with the corresponding untied run as closely as the architectures allow.

Early-stopping patience is set to 201 while the maximum training length is 200 epochs. Consequently, every successful run receives the same update budget, while `Validation layer_11_residual` still selects the checkpoint used for testing.

## One-time preparation

Run these short commands from the repository root. They verify the locked source assets, convert the published JSON split into the format expected by GridFM, and generate the nine resolved configs under the ignored `outputs/training-controls/` directory.

```bash
bash scripts/setup_vendor.sh
bash scripts/setup_env.sh cpu

.venv/bin/python scripts/download_assets.py \
  --lock assets/manifests/case118-assets.lock.json --scope full

.venv/bin/python scripts/prepare_training_controls.py
```

To inspect one resolved plan without starting training:

```bash
.venv/bin/python scripts/run_training_control.py \
  --architecture untied --seed 0 --dry-run
```

## Long-running training commands

The following commands are intentionally not executed by the repository workflow. On a single GPU, run them sequentially:

```bash
for seed in 0 1 2
do
  for architecture in untied tied_tail shared_graph_step_norms
  do
    .venv/bin/python scripts/run_training_control.py \
      --architecture "$architecture" \
      --seed "$seed" \
      --num-workers 16 \
      --mp-context spawn
  done
done
```

Each run writes its provenance and terminal status to `outputs/training-controls/runs/<architecture>/seed<seed>/manifest.json`. Model checkpoints, normalizer copies, split records, metrics, and MLflow metadata are stored under `outputs/training-controls/mlruns/`. The wrapper verifies all prepared hashes before training and records the best checkpoint hash and runtime after completion.

If a run has a manifest whose recorded status is `failed`, fix the underlying error and rerun the same command with `--retry-failed`. The wrapper archives the failed record as `manifest.attempt-<n>.json` before starting a new attempt. It refuses to retry manifests marked `running` or `complete`.

Use the same precision options for all nine runs. The default above is FP32. If TF32, bfloat16, compilation, worker count, or hardware changes, apply the change to all runs and report it as a separate protocol.

## Interpretation boundary

The primary comparison is final physical quality across three seeds. Parameter count and measured runtime should be reported separately: weight sharing reduces storage and trainable capacity but does not reduce the number of graph-operator calls at inference. Any untied–tied accuracy difference therefore mixes stage specialization with model capacity; the step-specific-LayerNorm control isolates only one part of that gap.

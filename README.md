# Inside GENCO: Stage-Specific Computation in a Learned Physics Solver

This project conducts an inference-only study on the computation inside [GENCO](https://arxiv.org/abs/2608.09921), a grid foundation model. GENCO shares its output heads, physics decoder, and residual-to-latent feedback, while each of its 12 corrections has a separate graph Transformer and bus/generator LayerNorm. We explore what those correction blocks do to get a feasible solution, and whether they perform interchangeable iterations or specialized refinement stages.

## Findings

### The model can look settled before the answer is ready

The model's internal patterns become similar across later steps, but its answer keeps getting closer to satisfying the grid's physical constraints.

![Step-wise physical refinement](figures/trajectories/physical_refinement_trajectory.png)

![Cross-correction representation similarity](figures/representations/cka_multiview_heatmaps.png)

### Small changes are still happening across the grid

Even when the overall internal pattern looks stable, many nodes can still receive small updates.

![Node activity across thresholds](figures/node-activity/node_activity_fraction.png)

### A quiet-looking step can still matter

We skip individual steps and check how the final answer changes. A step that barely changes the model's internal pattern can still help the solution.

![Representation change versus whole-correction bypass damage](figures/interventions/representation_change_vs_importance.png)

### Different steps are not simply interchangeable

We replace a step's learned processing block with one from another step. These swaps make the final answers worse on average in the model tested, supporting the idea that different steps have learned different roles.

![Operator substitution](figures/operators/operator_substitution_matrix.png)

We check how the solution develops, how nodes change, and what happens when steps are skipped across several released models. See the [research summary](docs/research-summary.md) for experiment details.

## Reproduce

Python 3.10–3.12 is supported. Here we set up CPU inference and download the pinned assets:

```bash
bash scripts/setup_vendor.sh
bash scripts/setup_env.sh cpu
.venv/bin/python scripts/download_assets.py --lock assets/manifests/case118-assets.lock.json --scope full
.venv/bin/python scripts/download_assets.py --lock assets/manifests/cross-network-assets.lock.json --scope full
```

See [reproducibility.md](docs/reproducibility.md) for experiment commands and [research-summary.md](docs/research-summary.md) for the evidence and figures.

GENCO checkpoints, datasets, and `gridfm-graphkit` come from the public releases pinned in the manifests and `vendor/graphkit-source.json`.

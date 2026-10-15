# Inside GENCO: Stage-Specific Computation in a Learned Physics Solver

This project looks inside [GENCO](https://arxiv.org/abs/2608.09921), a grid foundation model that refines a power-flow solution step by step. Each correction has its own graph-processing block, while the steps share the machinery that reads out the answer and feeds physical errors back into the model. We ask what each step contributes, and whether similar refinement could be learned by repeatedly reusing one block.

## Findings

### How a solution evloves throughout GENCO

The model's internal patterns become similar across later steps, but its answer keeps getting closer to satisfying the grid's physical constraints.

![Step-wise physical refinement](figures/trajectories/physical_refinement_trajectory.png)

![Cross-correction representation similarity](figures/representations/cka_multiview_heatmaps.png)

Even when the overall internal pattern looks stable, many nodes can still receive small updates.

![Node activity across thresholds](figures/node-activity/node_activity_fraction.png)

Besides, we skip individual steps and check how the final answer changes. A step that barely changes the model's internal pattern can still help the solution.

![Representation change versus whole-correction bypass damage](figures/interventions/representation_change_vs_importance.png)

### Different steps represent different roles

We replace a step's learned processing block with one from another step. These swaps make the final answers worse on average in the model tested, supporting the idea that different steps have learned different roles.

Swapping graph processing and normalization separately shows that both matter. In later steps, moving the learned pair together is often less disruptive than replacing either part alone.

![Operator substitution](figures/operators/operator_substitution_matrix.png)

The released model has learned different roles for its steps. But could a shared block learn to refine the changing state just as well? With limited compute, we trained small models with separate or shared blocks. They achieved similar performance and showed similar patterns of internal refinement, while the shared models used far fewer parameters.

![Representation similarity across separately trained architectures](figures/training-controls/cka_raw_best_seeds.png)

This makes recurrent sharing a promising direction to explore: keep refining the solution while reusing the same learned machinery. The [research summary](docs/research-summary.md) connects this result to looped Transformers and explains the experiments.

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

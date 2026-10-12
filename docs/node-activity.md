# Node-level state activity

## Definition

For operating condition `s`, correction `i`, and node `v`, activity is the relative hidden-state update

`r(s,i,v) = ||h_out(s,i,v) − h_in(s,i,v)||2 / (||h_in(s,i,v)||2 + 1e−12)`.

A node is active when `r > τ`. The analysis scans `τ = 0.01, 0.02, 0.05, 0.10, 0.20` across four checkpoints and 115 operating conditions per checkpoint. Correction 1 changes hidden width, so node-aligned measurements cover corrections 2–12.

## Activity over depth

![Activity fraction](../figures/node-activity/node_activity_fraction.png)

For case118 Base seed 0 at `τ=0.05`, active bus fractions at corrections 9–12 are `100.0%`, `98.9%`, `62.3%`, and `32.8%`. At correction 11, the per-condition median is `62.7%`, with a 5th–95th percentile range of `52.0%–69.5%`. At correction 12, the median is `33.9%`, with range `23.5%–39.2%`.

The generator trajectory stays active at larger magnitudes. Every generator update in corrections 7–12 exceeds `0.20` in the primary checkpoint.

## Threshold and checkpoint sensitivity

![Threshold heatmap](../figures/node-activity/node_activity_threshold_heatmap.png)

At `τ=0.05`, the active bus fractions at corrections 11 and 12 are:

| checkpoint | correction 11 | correction 12 |
| --- | ---: | ---: |
| case118 Base seed 0 | 62.3% | 32.8% |
| case118 Base seed 1 | 0.0% | 0.0% |
| case118 Small seed 0 | 0.0% | 0.0% |
| case30 Base seed 0 | 30.9% | 28.4% |

The scale varies by checkpoint. A calibrated threshold or budget-based gate can preserve a consistent compute allocation.

## Spatial structure

![Per-device update magnitudes](../figures/node-activity/node_activity_by_device.png)

The heatmap shows the median update of every stable device over 115 conditions. In the primary checkpoint, correction-11 bus medians span `0.0351–0.0681`; correction-12 medians span `0.0284–0.0624`. The ordering is smooth across bus identities and reproducible across the three case118 checkpoints.

The machine-readable aggregate is [node-activity-summary.csv](../results/node-activity-summary.csv). A full run writes per-node observations to `outputs/trajectory-analysis/<checkpoint>/node_updates.csv`.

## Compute implication

At correction 12 and `τ=0.05`, about two thirds of primary-checkpoint bus states fall into the reuse candidate set. A sparse implementation can retain prior states for those nodes and execute graph work for active nodes plus the edges required by their receptive fields. Evaluation should pair active-node and active-edge fractions with `ΔR∞`, voltage error, finite-output rate, and end-to-end latency.

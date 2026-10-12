"""Frozen compute policies, real step skipping, and policy-level statistics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import floor
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch_scatter import scatter_add


@dataclass(frozen=True)
class SkipPolicy:
    """One input-independent fixed-budget inference policy."""

    policy_id: str
    family: str
    budget: int
    step_numbers: tuple[int, ...]
    source: str
    policy_seed: int | None = None

    @property
    def layer_indices(self) -> frozenset[int]:
        return frozenset(step - 1 for step in self.step_numbers)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["step_numbers"] = list(self.step_numbers)
        value["layer_indices"] = sorted(self.layer_indices)
        value["executed_correction_steps"] = 12 - self.budget
        return value


def uniformly_spaced_steps(candidates: Sequence[int], budget: int) -> tuple[int, ...]:
    """Choose interior, evenly spaced positions without reading any scores.

    Positions divide the candidate interval into ``budget + 1`` equal gaps.
    Half positions use deterministic round-half-up rather than Python's
    round-to-even rule.
    """

    values = tuple(sorted(map(int, candidates)))
    if budget <= 0 or budget > len(values):
        raise ValueError("budget must be in 1..number of candidates")
    indices = [floor(index * (len(values) - 1) / (budget + 1) + 0.5) for index in range(1, budget + 1)]
    if len(set(indices)) != budget:
        raise ValueError("position rule produced duplicate candidates")
    return tuple(values[index] for index in indices)


def random_steps(candidates: Sequence[int], budget: int, seed: int) -> tuple[int, ...]:
    values = np.asarray(sorted(map(int, candidates)), dtype=np.int64)
    if budget <= 0 or budget > len(values):
        raise ValueError("budget must be in 1..number of candidates")
    return tuple(sorted(map(int, np.random.default_rng(int(seed)).choice(values, budget, replace=False))))


def build_policies(
    *,
    candidates: Sequence[int],
    budgets: Sequence[int],
    random_seeds: Sequence[int],
    local_scores: Mapping[int, float],
    global_rows: Iterable[Mapping[str, float]],
    global_constraints: Mapping[str, float],
) -> list[SkipPolicy]:
    """Build all frozen policies using exploration-side inputs only."""

    candidates = tuple(sorted(map(int, candidates)))
    budgets = tuple(sorted(map(int, budgets)))
    if len(candidates) != len(set(candidates)) or not candidates:
        raise ValueError("candidates must be unique and non-empty")
    if any(budget <= 0 or budget > len(candidates) for budget in budgets):
        raise ValueError("invalid policy budget")
    if set(local_scores) != set(candidates) or not all(np.isfinite(value) for value in local_scores.values()):
        raise ValueError("local scores must cover every candidate with finite values")

    eligible = []
    for raw in global_rows:
        row = dict(raw)
        step = int(row["correction_step_number"])
        if step not in candidates:
            continue
        if float(row["intervention_nonfinite_rate"]) > float(global_constraints["max_nonfinite_rate"]):
            continue
        if float(row["delta_r_inf_pu_100MVA_p95"]) > float(global_constraints["max_r_inf_p95"]):
            continue
        if abs(float(row["delta_vm_mae_pu_signed_mean"])) > float(global_constraints["max_abs_vm_mae_mean"]):
            continue
        if abs(float(row["delta_va_mae_rad_signed_mean"])) > float(global_constraints["max_abs_va_mae_mean"]):
            continue
        eligible.append(row)
    if len({int(row["correction_step_number"]) for row in eligible}) != len(eligible):
        raise ValueError("global ranking contains duplicate candidate steps")
    if len(eligible) < max(budgets):
        raise ValueError("global constraints leave too few eligible candidates")
    eligible.sort(key=lambda row: (
        float(row["delta_r_inf_pu_100MVA_positive_mean"]),
        float(row["delta_r_inf_pu_100MVA_p95"]),
        int(row["correction_step_number"]),
    ))
    local_rank = sorted(candidates, key=lambda step: (-float(local_scores[step]), step))

    policies = [SkipPolicy("complete", "complete", 0, (), "no bypass; frozen GENCO baseline")]
    for budget in budgets:
        for seed in random_seeds:
            chosen = random_steps(candidates, budget, int(seed))
            policies.append(SkipPolicy(
                f"random-s{int(seed)}-k{budget}", "random", budget, chosen,
                "uniform sample without replacement from frozen candidates", int(seed),
            ))
        position = uniformly_spaced_steps(candidates, budget)
        policies.append(SkipPolicy(
            f"position-k{budget}", "position", budget, position,
            "score-free interior equal-gap position rule",
        ))
        local = tuple(sorted(local_rank[:budget]))
        policies.append(SkipPolicy(
            f"local-cka-k{budget}", "local", budget, local,
            "highest exploration pooled bus whole-step CKA",
        ))
        global_steps = tuple(sorted(int(row["correction_step_number"]) for row in eligible[:budget]))
        policies.append(SkipPolicy(
            f"global-impact-k{budget}", "global", budget, global_steps,
            "lowest constrained exploration mean positive whole-step bypass damage",
        ))
    return policies


def combination_interaction(combo_delta: float, singleton_deltas: Sequence[float]) -> float | None:
    """Return combo damage minus the sum of its constituent single-step damages."""

    values = np.asarray([combo_delta, *singleton_deltas], dtype=float)
    return float(values[0] - values[1:].sum()) if np.isfinite(values).all() else None


class PolicyForwardAdapter:
    """Minimal frozen forward with a real branch around skipped correction steps.

    Unlike the traced adapter, this path does not clone hidden states or execute
    any graph convolution, solution decoder, physics decoder, PBRes, or encoded
    feedback operation for skipped steps.  It is therefore suitable for timing.
    """

    def __init__(self, task_or_model: torch.nn.Module):
        self.model = getattr(task_or_model, "model", task_or_model)
        required = (
            "layers", "input_proj_bus", "input_proj_gen", "input_proj_edge",
            "norms_bus", "norms_gen", "mlp_bus", "mlp_gen", "physics_mlp",
            "physics_decoder", "branch_flow_layer", "node_injection_layer",
            "node_residuals_layer",
        )
        missing = [name for name in required if not hasattr(self.model, name)]
        if missing:
            raise TypeError(f"unsupported model; missing {missing}")
        if getattr(self.model, "task", None) != "PowerFlow":
            raise ValueError("policy adapter supports the PowerFlow checkpoint only")

    def __call__(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict,
        edge_attr_dict: dict,
        mask_dict: dict[str, torch.Tensor],
        *,
        skip_layers: frozenset[int] = frozenset(),
    ) -> dict[str, torch.Tensor]:
        model = self.model
        layer_count = len(model.layers)
        if any(index <= 0 or index >= layer_count - 1 for index in skip_layers):
            raise ValueError("only intermediate correction layers may be skipped")

        h_bus = model.input_proj_bus(x_dict["bus"])
        h_gen = model.input_proj_gen(x_dict["gen"])
        num_bus = x_dict["bus"].size(0)
        _, gen_to_bus_index = edge_index_dict[("gen", "connected_to", "bus")]
        bus_edge_index = edge_index_dict[("bus", "connects", "bus")]
        bus_edge_attr = edge_attr_dict[("bus", "connects", "bus")]
        edge_attr_proj_dict = {
            key: model.input_proj_edge(value) if value is not None else None
            for key, value in edge_attr_dict.items()
        }
        from gridfm_graphkit.datasets.globals import VM_H, VA_H, PG_H

        bus_mask = mask_dict["bus"][:, VM_H : VA_H + 1]
        gen_mask = mask_dict["gen"][:, : PG_H + 1]
        bus_fixed = x_dict["bus"][:, VM_H : VA_H + 1]
        gen_fixed = x_dict["gen"][:, : PG_H + 1]
        output_temp = gen_temp = None

        for index, conv in enumerate(model.layers):
            if index in skip_layers:
                continue
            out = conv({"bus": h_bus, "gen": h_gen}, edge_index_dict, edge_attr_proj_dict)
            out_bus = model.activation(model.norms_bus[index](out["bus"]))
            out_gen = model.activation(model.norms_gen[index](out["gen"]))
            h_bus = h_bus + out_bus if out_bus.shape == h_bus.shape else out_bus
            h_gen = h_gen + out_gen if out_gen.shape == h_gen.shape else out_gen

            bus_temp = model.mlp_bus(h_bus)
            gen_temp = model.mlp_gen(h_gen)
            bus_temp = torch.where(bus_mask, bus_temp, bus_fixed)
            gen_temp = torch.where(gen_mask, gen_temp, gen_fixed)
            p_flow, q_flow = model.branch_flow_layer(bus_temp, bus_edge_index, bus_edge_attr)
            p_in, q_in = model.node_injection_layer(p_flow, q_flow, bus_edge_index, num_bus)
            agg_bus = scatter_add(gen_temp.squeeze(), gen_to_bus_index, dim=0, dim_size=num_bus)
            output_temp = model.physics_decoder(p_in, q_in, bus_temp, x_dict["bus"], agg_bus, mask_dict)
            residual_p, residual_q = model.node_residuals_layer(p_in, q_in, output_temp, x_dict["bus"])
            residual = torch.stack([residual_p, residual_q], dim=-1)
            h_bus = h_bus + model.physics_mlp(residual)

        if output_temp is None or gen_temp is None:
            raise RuntimeError("no correction layer produced a prediction")
        return {"bus": output_temp, "gen": gen_temp}

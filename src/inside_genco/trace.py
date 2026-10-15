"""A thin, research-side traced forward for the frozen GENCO PF model.

The execution order mirrors ``gridfm_graphkit`` commit
95b90eecc7f5c2c86a9dd776f0c3f40434def22f.  It calls the checkpoint's own
modules and does not patch the vendor snapshot. Equivalence checks compare each
observable stage against read-only hooks on the original forward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch_scatter import scatter_add

from .interventions import InterventionPlan


def _copy(tensor: torch.Tensor, to_cpu: bool) -> torch.Tensor:
    result = tensor.detach().clone()
    return result.cpu() if to_cpu else result


def _operator_modules(
    model: torch.nn.Module, target_index: int, plan: InterventionPlan
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, int]:
    """Resolve graph and normalization sources for joint or component swaps."""

    source_index = int(plan.operator_sources.get(target_index, target_index))
    graph_index = target_index if plan.operator_component == "norm" else source_index
    norm_index = target_index if plan.operator_component == "graph" else source_index
    return (
        model.layers[graph_index],
        model.norms_bus[norm_index],
        model.norms_gen[norm_index],
        source_index,
    )


@dataclass(frozen=True)
class LayerTrace:
    layer_index: int
    display_layer: int
    h_in_bus: torch.Tensor
    h_in_gen: torch.Tensor
    h_graph_bus: torch.Tensor
    h_graph_gen: torch.Tensor
    x_bus: torch.Tensor | None
    x_gen: torch.Tensor | None
    p_flow: torch.Tensor | None
    q_flow: torch.Tensor | None
    residual_bus: torch.Tensor | None
    feedback_bus: torch.Tensor | None
    h_out_bus: torch.Tensor
    h_out_gen: torch.Tensor
    skipped: bool


@dataclass(frozen=True)
class ForwardTrace:
    output: dict[str, torch.Tensor]
    layers: tuple[LayerTrace, ...]
    layer_residuals: dict[int, torch.Tensor]


@dataclass(frozen=True)
class HookObservation:
    output: dict[str, torch.Tensor]
    h_graph_bus: tuple[torch.Tensor, ...]
    h_graph_gen: tuple[torch.Tensor, ...]
    residual_bus: tuple[torch.Tensor, ...]
    feedback_bus: tuple[torch.Tensor, ...]
    layer_residuals: dict[int, torch.Tensor]


class TraceForwardAdapter:
    """Trace/intervene while reusing all modules from one loaded GENCO model."""

    def __init__(self, task_or_model: torch.nn.Module, *, snapshot_to_cpu: bool = True):
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
            raise ValueError("trace adapter currently supports the PowerFlow checkpoint only")
        self.snapshot_to_cpu = snapshot_to_cpu

    def __call__(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict,
        edge_attr_dict: dict,
        mask_dict: dict[str, torch.Tensor],
        *,
        intervention: InterventionPlan | None = None,
    ) -> ForwardTrace:
        model = self.model
        plan = intervention or InterventionPlan()
        plan.validate(len(model.layers))
        model.layer_residuals = {}

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

        # Imported lazily so synthetic unit tests need only the frozen model API.
        from gridfm_graphkit.datasets.globals import VM_H, VA_H, PG_H

        bus_mask = mask_dict["bus"][:, VM_H : VA_H + 1]
        gen_mask = mask_dict["gen"][:, : PG_H + 1]
        bus_fixed = x_dict["bus"][:, VM_H : VA_H + 1]
        gen_fixed = x_dict["gen"][:, : PG_H + 1]
        traces: list[LayerTrace] = []
        output_temp = gen_temp = None

        for index in range(len(model.layers)):
            h_in_bus, h_in_gen = h_bus, h_gen
            if index in plan.skip_layers:
                h_bus, h_gen = h_in_bus, h_in_gen
                traces.append(
                    LayerTrace(
                        index, index + 1,
                        _copy(h_in_bus, self.snapshot_to_cpu), _copy(h_in_gen, self.snapshot_to_cpu),
                        _copy(h_bus, self.snapshot_to_cpu), _copy(h_gen, self.snapshot_to_cpu),
                        None, None, None, None, None, None,
                        _copy(h_bus, self.snapshot_to_cpu), _copy(h_gen, self.snapshot_to_cpu),
                        True,
                    )
                )
                continue

            conv, norm_bus, norm_gen, _source_index = _operator_modules(
                model, index, plan
            )
            out = conv({"bus": h_bus, "gen": h_gen}, edge_index_dict, edge_attr_proj_dict)
            out_bus = model.activation(norm_bus(out["bus"]))
            out_gen = model.activation(norm_gen(out["gen"]))
            h_graph_bus = h_bus + out_bus if out_bus.shape == h_bus.shape else out_bus
            h_graph_gen = h_gen + out_gen if out_gen.shape == h_gen.shape else out_gen

            bus_temp = model.mlp_bus(h_graph_bus)
            gen_temp = model.mlp_gen(h_graph_gen)
            bus_temp = torch.where(bus_mask, bus_temp, bus_fixed)
            gen_temp = torch.where(gen_mask, gen_temp, gen_fixed)
            p_flow, q_flow = model.branch_flow_layer(bus_temp, bus_edge_index, bus_edge_attr)
            p_in, q_in = model.node_injection_layer(p_flow, q_flow, bus_edge_index, num_bus)
            agg_bus = scatter_add(gen_temp.squeeze(), gen_to_bus_index, dim=0, dim_size=num_bus)
            output_temp = model.physics_decoder(
                p_in, q_in, bus_temp, x_dict["bus"], agg_bus, mask_dict,
            )
            residual_p, residual_q = model.node_residuals_layer(
                p_in, q_in, output_temp, x_dict["bus"],
            )
            residual = torch.stack([residual_p, residual_q], dim=-1)
            model.layer_residuals[index] = torch.linalg.norm(residual, dim=-1).mean()
            feedback = model.physics_mlp(residual)
            h_out_bus = h_graph_bus + feedback
            h_out_gen = h_graph_gen

            traces.append(
                LayerTrace(
                    index, index + 1,
                    _copy(h_in_bus, self.snapshot_to_cpu), _copy(h_in_gen, self.snapshot_to_cpu),
                    _copy(h_graph_bus, self.snapshot_to_cpu), _copy(h_graph_gen, self.snapshot_to_cpu),
                    _copy(output_temp, self.snapshot_to_cpu), _copy(gen_temp, self.snapshot_to_cpu),
                    _copy(p_flow, self.snapshot_to_cpu), _copy(q_flow, self.snapshot_to_cpu),
                    _copy(residual, self.snapshot_to_cpu), _copy(feedback, self.snapshot_to_cpu),
                    _copy(h_out_bus, self.snapshot_to_cpu), _copy(h_out_gen, self.snapshot_to_cpu),
                    False,
                )
            )
            h_bus, h_gen = h_out_bus, h_out_gen

        if output_temp is None or gen_temp is None:
            raise RuntimeError("no layer produced a prediction")
        output = {"bus": output_temp, "gen": gen_temp}
        return ForwardTrace(
            output=output,
            layers=tuple(traces),
            layer_residuals={key: _copy(value, self.snapshot_to_cpu) for key, value in model.layer_residuals.items()},
        )


def observe_with_hooks(
    task_or_model: torch.nn.Module,
    x_dict: dict[str, torch.Tensor],
    edge_index_dict: dict,
    edge_attr_dict: dict,
    mask_dict: dict[str, torch.Tensor],
    *,
    snapshot_to_cpu: bool = True,
) -> HookObservation:
    """Run the untouched forward with read-only hooks on shared modules."""

    model = getattr(task_or_model, "model", task_or_model)
    captured: dict[str, list[torch.Tensor]] = {
        "h_graph_bus": [], "h_graph_gen": [], "residual_bus": [], "feedback_bus": [],
    }
    handles = [
        model.mlp_bus.register_forward_pre_hook(
            lambda _module, args: captured["h_graph_bus"].append(_copy(args[0], snapshot_to_cpu))
        ),
        model.mlp_gen.register_forward_pre_hook(
            lambda _module, args: captured["h_graph_gen"].append(_copy(args[0], snapshot_to_cpu))
        ),
        model.physics_mlp.register_forward_pre_hook(
            lambda _module, args: captured["residual_bus"].append(_copy(args[0], snapshot_to_cpu))
        ),
        model.physics_mlp.register_forward_hook(
            lambda _module, _args, output: captured["feedback_bus"].append(_copy(output, snapshot_to_cpu))
        ),
    ]
    try:
        output = task_or_model(x_dict, edge_index_dict, edge_attr_dict, mask_dict)
    finally:
        for handle in handles:
            handle.remove()
    expected = len(model.layers)
    if any(len(values) != expected for values in captured.values()):
        raise RuntimeError(f"shared-module hook call count mismatch: { {k: len(v) for k, v in captured.items()} }")
    return HookObservation(
        output={key: _copy(value, snapshot_to_cpu) for key, value in output.items()},
        h_graph_bus=tuple(captured["h_graph_bus"]),
        h_graph_gen=tuple(captured["h_graph_gen"]),
        residual_bus=tuple(captured["residual_bus"]),
        feedback_bus=tuple(captured["feedback_bus"]),
        layer_residuals={key: _copy(value, snapshot_to_cpu) for key, value in model.layer_residuals.items()},
    )

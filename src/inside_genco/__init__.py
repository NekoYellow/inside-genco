"""Observation, intervention, and compute-allocation tools for GENCO."""

from .interventions import InterventionPlan, TensorEdit, stable_relative_perturbation
from .similarity import CKAResult, linear_cka
from .trace import ForwardTrace, LayerTrace, TraceForwardAdapter, observe_with_hooks
from .trajectory_metrics import (
    LinearCKAAccumulator,
    LocalStepMetrics,
    first_threshold_crossing,
    local_step_metrics,
    node_relative_l2_updates,
)

__all__ = [
    "CKAResult",
    "ForwardTrace",
    "InterventionPlan",
    "LayerTrace",
    "LinearCKAAccumulator",
    "LocalStepMetrics",
    "TensorEdit",
    "TraceForwardAdapter",
    "first_threshold_crossing",
    "linear_cka",
    "local_step_metrics",
    "node_relative_l2_updates",
    "observe_with_hooks",
    "stable_relative_perturbation",
]

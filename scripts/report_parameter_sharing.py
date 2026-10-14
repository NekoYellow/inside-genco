#!/usr/bin/env python3
"""Report GENCO parameter counts under recurrent-tail sharing controls."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor/gridfm-graphkit"))

from baseline import load_model
from run_trajectory_analysis import load_variant
from inside_genco.tied_tail import count_unique_parameters, tie_recurrent_tail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="case118_base_seed0")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/checkpoints.yaml")
    args = parser.parse_args()

    variant = load_variant(args.config, args.variant)
    _config, _normalizer, _stats, task = load_model(
        "cpu", variant["model_dir"], variant["network"]
    )
    original = count_unique_parameters(task.model)
    variants = []
    for share_norms in (True, False):
        model = copy.deepcopy(task.model)
        report = tie_recurrent_tail(model, share_norms=share_norms)
        variants.append(
            {
                **report.to_dict(),
                "parameter_fraction_of_untied": (
                    report.unique_parameters_after / original
                ),
                "parameter_reduction_factor": (
                    original / report.unique_parameters_after
                ),
            }
        )
    result = {
        "schema_version": 1,
        "variant": args.variant,
        "untied_unique_parameters": original,
        "controls": variants,
        "scope": (
            "architecture-only parameter accounting; no tied checkpoint is evaluated"
        ),
    }
    output = ROOT / "results/parameter-sharing.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

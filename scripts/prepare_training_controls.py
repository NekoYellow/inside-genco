#!/usr/bin/env python3
"""Materialize one fixed split and nine matched GENCO training configs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def validate_split(split: dict[str, list[int]]) -> None:
    if set(split) != {"train", "val", "test"}:
        raise ValueError("split JSON must contain exactly train, val, and test")
    seen: set[int] = set()
    for name in ("train", "val", "test"):
        values = split[name]
        if not values or any(type(value) is not int or value < 0 for value in values):
            raise ValueError(f"invalid {name} scenario IDs")
        if len(values) != len(set(values)) or seen.intersection(values):
            raise ValueError("split IDs must be unique and disjoint")
        seen.update(values)


def build_config(
    source: dict,
    *,
    model_type: str,
    seed: int,
    split_directory: str,
    epochs: int,
    patience: int,
) -> dict:
    config = copy.deepcopy(source)
    config["seed"] = int(seed)
    config["model"]["type"] = model_type
    config["data"].pop("split_by_load_scenario_idx", None)
    config["data"]["split_from_existing_files"] = split_directory
    config["training"]["epochs"] = int(epochs)
    config["callbacks"]["patience"] = int(patience)
    return config


def validate_invariants(config: dict, expected: dict) -> None:
    checks = {
        "corrections": config["model"]["num_layers"],
        "hidden_size": config["model"]["hidden_size"],
        "attention_heads": config["model"]["attention_head"],
        "batch_size": config["training"]["batch_size"],
        "losses": config["training"]["losses"],
        "loss_weights": config["training"]["loss_weights"],
        "learning_rate": config["optimizer"]["learning_rate"],
        "beta1": config["optimizer"]["beta1"],
        "beta2": config["optimizer"]["beta2"],
        "lr_decay": config["optimizer"]["lr_decay"],
        "lr_patience": config["optimizer"]["lr_patience"],
    }
    mismatches = {
        key: {"expected": expected[key], "actual": value}
        for key, value in checks.items()
        if value != expected[key]
    }
    if mismatches:
        raise ValueError(f"source config violates training protocol: {mismatches}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/training-controls.yaml"
    )
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = yaml.safe_load(protocol_path.read_text())
    source_path = resolve(ROOT, protocol["source_config"])
    split_json_path = resolve(ROOT, protocol["source_split_json"])
    normalizer_path = resolve(ROOT, protocol["normalizer_stats"])
    for path in (source_path, split_json_path, normalizer_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    source = yaml.safe_load(source_path.read_text())
    validate_invariants(source, protocol["invariants"])
    split = json.loads(split_json_path.read_text())
    validate_split(split)

    output_root = resolve(ROOT, protocol["output_root"])
    split_directory = output_root / "splits" / "case118_published_seed0"
    config_directory = output_root / "configs"
    split_directory.mkdir(parents=True, exist_ok=True)
    config_directory.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        torch.save(torch.tensor(split[name], dtype=torch.long), split_directory / f"{name}.pt")

    relative_split = str(split_directory.relative_to(ROOT))
    generated = []
    for architecture, specification in protocol["architectures"].items():
        for seed in protocol["seeds"]:
            config = build_config(
                source,
                model_type=specification["model_type"],
                seed=int(seed),
                split_directory=relative_split,
                epochs=int(protocol["budget"]["epochs"]),
                patience=int(protocol["budget"]["early_stopping_patience"]),
            )
            validate_invariants(config, protocol["invariants"])
            destination = config_directory / f"{architecture}_seed{seed}.yaml"
            destination.write_text(yaml.safe_dump(config, sort_keys=False))
            generated.append(
                {
                    "architecture": architecture,
                    "model_type": specification["model_type"],
                    "seed": int(seed),
                    "config": str(destination.relative_to(ROOT)),
                    "config_sha256": sha256(destination),
                }
            )

    manifest = {
        "schema_version": 1,
        "status": "prepared",
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": sha256(protocol_path),
        "source_config": str(source_path.relative_to(ROOT)),
        "source_config_sha256": sha256(source_path),
        "source_split_json": str(split_json_path.relative_to(ROOT)),
        "source_split_json_sha256": sha256(split_json_path),
        "normalizer_stats": str(normalizer_path.relative_to(ROOT)),
        "normalizer_stats_sha256": sha256(normalizer_path),
        "split_counts": {name: len(split[name]) for name in ("train", "val", "test")},
        "split_files": {
            name: {
                "path": str((split_directory / f"{name}.pt").relative_to(ROOT)),
                "sha256": sha256(split_directory / f"{name}.pt"),
            }
            for name in ("train", "val", "test")
        },
        "checkpoint_selection": protocol["checkpoint_selection"],
        "budget": protocol["budget"],
        "runs": generated,
    }
    manifest_path = output_root / "prepared-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

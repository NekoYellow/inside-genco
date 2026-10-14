#!/usr/bin/env python3
"""Run one matched GENCO architecture control with a provenance manifest."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor/gridfm-graphkit"))

ARCHITECTURES = ("untied", "tied_tail", "shared_graph_step_norms")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_run_directory(
    run_directory: Path, manifest_path: Path, *, retry_failed: bool, resume_interrupted: bool = False
) -> Path | None:
    """Create a fresh run directory or archive one failed attempt for a retry."""
    if not manifest_path.exists():
        if run_directory.exists():
            raise FileExistsError(
                f"run directory exists without a manifest: {run_directory}"
            )
        run_directory.mkdir(parents=True)
        return None

    if not retry_failed and not resume_interrupted:
        raise FileExistsError(
            f"run manifest already exists: {manifest_path}; pass --retry-failed "
            "only if its status is failed"
        )

    previous = json.loads(manifest_path.read_text())
    allowed = ("running", "failed") if resume_interrupted else ("failed",)
    if previous.get("status") not in allowed:
        raise FileExistsError(
            f"refusing to retry run with status {previous.get('status')!r}: "
            f"{manifest_path}"
        )

    attempt = 1
    while True:
        archived = manifest_path.with_name(f"manifest.attempt-{attempt}.json")
        if not archived.exists():
            break
        attempt += 1
    manifest_path.replace(archived)
    return archived


def inspect_resume_checkpoint(path: Path, completed_epochs: int, target_epochs: int) -> dict:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not all(checkpoint.get(key) for key in ("state_dict", "optimizer_states", "lr_schedulers", "loops")):
        raise ValueError("resume requires a full training checkpoint, not weights only")
    if checkpoint.get("epoch") != completed_epochs - 1:
        raise ValueError("checkpoint epoch differs from the completed run")
    if target_epochs <= completed_epochs or checkpoint.get("global_step", 0) <= 0:
        raise ValueError("resume target must extend a completed training run")
    return {"completed_epochs": completed_epochs, "global_step": checkpoint["global_step"]}


def validate_resume_config(parent: dict, source: dict) -> None:
    comparable = copy.deepcopy(parent)
    comparable["training"]["epochs"] = source["training"]["epochs"]
    comparable["callbacks"]["patience"] = source["callbacks"]["patience"]
    if comparable != source:
        raise ValueError("resume config differs beyond epoch budget and patience")


def invoke_training(namespace, resume: dict | None) -> None:
    from gridfm_graphkit import cli

    if resume is None:
        cli.main_cli(namespace)
        return
    from gridfm_graphkit.training.callbacks import SaveBestModelStateDict, best_model_artifact_path

    original_fit = cli.L.Trainer.fit

    def resumed_fit(trainer, *args, **kwargs):
        # The upstream best-weights callback does not serialize its best score.
        # Carry the old best forward so a worse continuation cannot replace it.
        for callback in trainer.callbacks:
            if isinstance(callback, SaveBestModelStateDict):
                callback.best_score = resume["best_validation_score"]
                destination = Path(best_model_artifact_path(trainer.logger, callback.filename))
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / resume["best_checkpoint"], destination)
        trainer.num_sanity_val_steps = 0
        kwargs["ckpt_path"] = str(ROOT / resume["last_checkpoint"])
        print(f"Resuming after epoch {resume['completed_epochs']} at global step {resume['global_step']}", flush=True)
        return original_fit(trainer, *args, **kwargs)

    with patch.object(cli.L.Trainer, "fit", resumed_fit):
        cli.main_cli(namespace)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--mp-context", choices=("spawn", "fork", "forkserver"))
    parser.add_argument("--compile", choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"))
    parser.add_argument("--bfloat16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="archive and retry an existing manifest only when its status is failed",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-to-epoch", type=int, help="resume the completed base run to this total epoch budget")
    parser.add_argument("--resume-from-epoch", type=int, help="use a completed continuation at this epoch instead of the base run")
    parser.add_argument("--resume-interrupted-checkpoint", type=Path, help="full last.ckpt from an interrupted target run; ensure its process is stopped")
    args = parser.parse_args()
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.resume_from_epoch is not None and args.resume_to_epoch is None:
        parser.error("--resume-from-epoch requires --resume-to-epoch")
    if args.resume_interrupted_checkpoint and (args.resume_to_epoch is None or args.resume_from_epoch is None):
        parser.error("interrupted recovery requires --resume-from-epoch (saved epoch) and --resume-to-epoch")

    protocol_path = ROOT / "configs/training-controls.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    if args.seed not in protocol["seeds"]:
        parser.error(f"seed must be one of {protocol['seeds']}")
    prepared_path = ROOT / protocol["output_root"] / "prepared-manifest.json"
    if not prepared_path.is_file():
        raise FileNotFoundError(
            "training controls are not prepared; run scripts/prepare_training_controls.py"
        )
    prepared = json.loads(prepared_path.read_text())
    if sha256(protocol_path) != prepared["protocol_sha256"]:
        raise ValueError("protocol differs from the prepared manifest")
    run_spec = next(
        run
        for run in prepared["runs"]
        if run["architecture"] == args.architecture and run["seed"] == args.seed
    )
    config_path = ROOT / run_spec["config"]
    if sha256(config_path) != run_spec["config_sha256"]:
        raise ValueError("resolved training config hash differs from prepared manifest")
    normalizer_path = ROOT / prepared["normalizer_stats"]
    if sha256(normalizer_path) != prepared["normalizer_stats_sha256"]:
        raise ValueError("normalizer hash differs from prepared manifest")
    for split in prepared["split_files"].values():
        path = ROOT / split["path"]
        if sha256(path) != split["sha256"]:
            raise ValueError(f"fixed split hash differs: {path}")

    output_root = ROOT / protocol["output_root"]
    log_dir = output_root / "mlruns"
    run_directory = output_root / "runs" / args.architecture / f"seed{args.seed}"
    manifest_path = run_directory / "manifest.json"
    plan = {
        "architecture": args.architecture,
        "seed": args.seed,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "fixed_split_manifest_sha256": sha256(prepared_path),
        "normalizer_stats": str(normalizer_path.relative_to(ROOT)),
        "normalizer_stats_sha256": sha256(normalizer_path),
        "data_path": protocol["data_path"],
        "epochs": protocol["budget"]["epochs"],
        "checkpoint_selection": protocol["checkpoint_selection"],
        "losses": protocol["invariants"]["losses"],
        "loss_weights": protocol["invariants"]["loss_weights"],
        "optimizer": {
            key: protocol["invariants"][key]
            for key in ("optimizer", "learning_rate", "beta1", "beta2", "lr_decay", "lr_patience")
        },
        "precision": "bfloat16" if args.bfloat16 else "float32",
        "tf32": bool(args.tf32),
        "compile": args.compile,
        "num_workers": args.num_workers,
        "mp_context": args.mp_context,
        "log_dir": str(log_dir.relative_to(ROOT)),
        "run_name": f"{args.architecture}-seed{args.seed}",
    }
    resume = None
    continuation_config = None
    if args.resume_to_epoch is not None:
        parent_path = manifest_path
        if args.resume_from_epoch is not None:
            parent_path = output_root / f"resume-to-{args.resume_from_epoch}" / "runs" / args.architecture / f"seed{args.seed}" / "manifest.json"
        if args.resume_interrupted_checkpoint:
            parent_path = output_root / f"resume-to-{args.resume_to_epoch}" / "runs" / args.architecture / f"seed{args.seed}" / "manifest.json"
        parent = json.loads(parent_path.read_text())
        allowed_status = ("running", "failed") if args.resume_interrupted_checkpoint else ("complete",)
        if parent["status"] not in allowed_status:
            raise ValueError("resume requires a completed parent-run manifest")
        expected_epoch = args.resume_from_epoch if args.resume_from_epoch is not None else protocol["budget"]["epochs"]
        if parent["plan"]["epochs"] != (args.resume_to_epoch if args.resume_interrupted_checkpoint else expected_epoch):
            raise ValueError("parent run does not match the requested starting epoch")
        parent_config = ROOT / parent["plan"]["config"]
        if sha256(parent_config) != parent["plan"]["config_sha256"]:
            raise ValueError("parent training config hash differs")
        validate_resume_config(yaml.safe_load(parent_config.read_text()), yaml.safe_load(config_path.read_text()))
        for key in ("architecture", "seed", "fixed_split_manifest_sha256",
                    "normalizer_stats_sha256", "precision", "tf32", "compile"):
            if parent["plan"][key] != plan[key]:
                raise ValueError(f"resume would change the original training setup: {key}")
        if args.resume_interrupted_checkpoint:
            last = (ROOT / args.resume_interrupted_checkpoint).resolve()
            run_root = last.parents[1]
            if last.name != "last.ckpt" or last.parent.name != "checkpoints" or not last.is_relative_to((ROOT / parent["plan"]["log_dir"]).resolve()):
                raise ValueError("interrupted checkpoint is outside the target run logs")
            if (run_root / "tags/mlflow.runName").read_text().strip() != parent["plan"]["run_name"]:
                raise ValueError("interrupted checkpoint belongs to a different run")
            candidates = list((run_root / "artifacts").rglob("best_model_state_dict.pt"))
            if len(candidates) != 1:
                raise ValueError("expected one best checkpoint in interrupted run")
            best = candidates[0]
        else:
            best = ROOT / parent["best_checkpoint"]
            last = best.parents[2] / "checkpoints/last.ckpt"
        if not args.resume_interrupted_checkpoint and sha256(best) != parent["best_checkpoint_sha256"]:
            raise ValueError("original best-checkpoint hash differs")
        if "last_checkpoint_sha256" in parent and sha256(last) != parent["last_checkpoint_sha256"]:
            raise ValueError("parent last-checkpoint hash differs")
        resume = inspect_resume_checkpoint(last, expected_epoch, args.resume_to_epoch)
        scores = best.parents[2] / "metrics" / protocol["checkpoint_selection"]["monitor"]
        resume.update(
            parent_manifest=str(parent_path.relative_to(ROOT)),
            parent_manifest_sha256=sha256(parent_path),
            last_checkpoint=str(last.relative_to(ROOT)),
            last_checkpoint_sha256=sha256(last),
            best_checkpoint=str(best.relative_to(ROOT)),
            best_checkpoint_sha256=sha256(best),
            best_validation_score=min(
                min(float(line.split()[1]) for line in scores.read_text().splitlines()),
                parent["plan"].get("resume", {}).get("best_validation_score", float("inf")),
            ),
        )
        continuation = yaml.safe_load(config_path.read_text())
        continuation["training"]["epochs"] = args.resume_to_epoch
        continuation["callbacks"]["patience"] = args.resume_to_epoch + 1
        continuation_config = yaml.safe_dump(continuation, sort_keys=False)
        continuation_root = output_root / f"resume-to-{args.resume_to_epoch}"
        log_dir = continuation_root / "mlruns"
        run_directory = continuation_root / "runs" / args.architecture / f"seed{args.seed}"
        manifest_path = run_directory / "manifest.json"
        config_path = run_directory / "config.yaml"
        plan.update(
            config=str(config_path.relative_to(ROOT)),
            config_sha256=hashlib.sha256(continuation_config.encode()).hexdigest(),
            epochs=args.resume_to_epoch,
            resume=resume,
            log_dir=str(log_dir.relative_to(ROOT)),
            run_name=f"{args.architecture}-seed{args.seed}-to{args.resume_to_epoch}",
        )
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    previous_attempt = prepare_run_directory(
        run_directory, manifest_path, retry_failed=args.retry_failed,
        resume_interrupted=bool(args.resume_interrupted_checkpoint),
    )
    if args.resume_interrupted_checkpoint and previous_attempt is not None:
        resume["parent_manifest"] = str(previous_attempt.relative_to(ROOT))
    if continuation_config is not None:
        config_path.write_text(continuation_config)
    started = time.perf_counter()
    manifest = {
        "schema_version": 1,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "research_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "plan": plan,
    }
    if previous_attempt is not None:
        manifest["retry_of"] = str(previous_attempt.relative_to(ROOT))

    def save() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    before = set(log_dir.rglob("best_model_state_dict.pt")) if log_dir.exists() else set()
    try:
        from argparse import Namespace

        namespace = Namespace(
            command="train",
            config=str(config_path),
            exp_name=protocol["name"],
            run_name=plan["run_name"],
            log_dir=str(log_dir),
            data_path=str(ROOT / protocol["data_path"]),
            normalizer_stats=str(normalizer_path),
            compile=args.compile,
            bfloat16=args.bfloat16,
            tf32=args.tf32,
            dataset_wrapper=None,
            plugins=["inside_genco.tied_tail_plugin"],
            num_workers=args.num_workers,
            dataset_wrapper_cache_dir=None,
            profiler=None,
            compute_dc_ac_metrics=False,
            report_performance=True,
            mp_context=args.mp_context,
        )
        invoke_training(namespace, resume)
        after = set(log_dir.rglob("best_model_state_dict.pt"))
        created = sorted(after - before)
        if len(created) != 1:
            raise RuntimeError(f"expected one new best checkpoint, found {len(created)}")
        checkpoint = created[0]
        if resume is not None:
            last_checkpoint = checkpoint.parents[2] / "checkpoints/last.ckpt"
            completed = inspect_resume_checkpoint(last_checkpoint, plan["epochs"], plan["epochs"] + 1)
            if completed["global_step"] <= resume["global_step"]:
                raise RuntimeError("continuation made no optimizer-step progress")
            manifest.update(
                last_checkpoint=str(last_checkpoint.relative_to(ROOT)),
                last_checkpoint_sha256=sha256(last_checkpoint),
                final_global_step=completed["global_step"],
            )
        manifest.update(
            status="complete",
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            runtime_seconds=time.perf_counter() - started,
            best_checkpoint=str(checkpoint.relative_to(ROOT)),
            best_checkpoint_sha256=sha256(checkpoint),
        )
        save()
    except BaseException as error:
        manifest.update(
            status="failed",
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            runtime_seconds=time.perf_counter() - started,
            error={"type": type(error).__name__, "message": str(error)},
        )
        save()
        raise


if __name__ == "__main__":
    main()

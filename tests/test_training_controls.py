from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_training_controls import build_config, sample_grouped_split, validate_invariants, validate_split
from run_training_control import inspect_resume_checkpoint, prepare_run_directory, validate_resume_config


class TrainingControlTests(unittest.TestCase):
    def test_interrupted_recovery_archives_running_manifest(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"status": "running"}))
            archived = prepare_run_directory(root, manifest, retry_failed=False, resume_interrupted=True)
            self.assertEqual(json.loads(archived.read_text())["status"], "running")
            self.assertFalse(manifest.exists())
            manifest.write_text(json.dumps({"status": "complete"}))
            with self.assertRaises(FileExistsError):
                prepare_run_directory(root, manifest, retry_failed=False, resume_interrupted=True)

    def test_chained_resume_preserves_training_setup(self) -> None:
        import copy

        source = self.source
        parent = copy.deepcopy(source)
        parent["training"]["epochs"] = 70
        parent["callbacks"]["patience"] = 71
        validate_resume_config(parent, source)
        parent["optimizer"]["learning_rate"] *= 2
        with self.assertRaisesRegex(ValueError, "differs beyond"):
            validate_resume_config(parent, source)

    def test_resume_requires_full_state_and_extends_total_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.ckpt"
            checkpoint = dict(state_dict={"weight": torch.ones(1)}, optimizer_states=[{}],
                              lr_schedulers=[{}], loops={"fit_loop": {}}, epoch=49, global_step=13700)
            torch.save(checkpoint, path)
            self.assertEqual(inspect_resume_checkpoint(path, 50, 70),
                             {"completed_epochs": 50, "global_step": 13700})
            with self.assertRaisesRegex(ValueError, "extend"):
                inspect_resume_checkpoint(path, 50, 50)
            with self.assertRaisesRegex(ValueError, "epoch differs"):
                inspect_resume_checkpoint(path, 49, 70)
            checkpoint.pop("optimizer_states")
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "full training checkpoint"):
                inspect_resume_checkpoint(path, 50, 70)

    def setUp(self) -> None:
        self.protocol = yaml.safe_load((ROOT / "configs/training-controls.yaml").read_text())
        source_path = ROOT / self.protocol["source_config"]
        self.source = yaml.safe_load(source_path.read_text())

    def test_all_architectures_preserve_protocol_invariants(self) -> None:
        for architecture in self.protocol["architectures"].values():
            for seed in self.protocol["seeds"]:
                config = build_config(
                    self.source,
                    model_type=architecture["model_type"],
                    seed=seed,
                    split_directory="fixed/split",
                    epochs=self.protocol["budget"]["epochs"],
                    patience=self.protocol["budget"]["early_stopping_patience"],
                )
                validate_invariants(config, self.protocol["invariants"])
                self.assertEqual(config["seed"], seed)
                self.assertEqual(config["training"]["epochs"], 50)
                self.assertEqual(config["callbacks"]["patience"], 51)
                self.assertEqual(config["data"]["split_from_existing_files"], "fixed/split")
                self.assertNotIn("split_by_load_scenario_idx", config["data"])

    def test_split_validation_rejects_overlap(self) -> None:
        validate_split({"train": [0, 1], "val": [2], "test": [3]})
        with self.assertRaisesRegex(ValueError, "unique and disjoint"):
            validate_split({"train": [0, 1], "val": [1], "test": [3]})

    def test_group_sampling_is_fixed_and_preserves_entire_groups(self) -> None:
        split = {"train": [0, 1, 2, 3], "val": [4, 5, 6, 7], "test": [8, 9]}
        groups = {sid: sid // 2 for sid in range(10)}
        subset = dict(seed=42, train_group_fraction=.5, val_group_fraction=.5)
        sampled, counts = sample_grouped_split(split, groups, subset)
        reordered = {name: list(reversed(ids)) for name, ids in split.items()}
        self.assertEqual(sampled, sample_grouped_split(reordered, groups, subset)[0])
        self.assertEqual(sampled["test"], split["test"])
        for name in ("train", "val"):
            self.assertEqual(len(sampled[name]), 2)
            self.assertEqual(len({groups[sid] for sid in sampled[name]}), 1)
            self.assertEqual(counts[name], {"original": 2, "selected": 1})
        groups[4] = groups[0]
        with self.assertRaisesRegex(ValueError, "load groups overlap"):
            sample_grouped_split(split, groups, subset)

    def test_group_sampling_rejects_missing_metadata_and_invalid_fraction(self) -> None:
        split = {"train": [0], "val": [1], "test": [2]}
        subset = dict(seed=42, train_group_fraction=0, val_group_fraction=.5)
        with self.assertRaisesRegex(ValueError, "does not cover"):
            sample_grouped_split(split, {0: 0}, subset)
        with self.assertRaisesRegex(ValueError, "fraction"):
            sample_grouped_split(split, {0: 0, 1: 1, 2: 2}, subset)

    def test_retry_archives_only_failed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory) / "seed0"
            run_directory.mkdir()
            manifest_path = run_directory / "manifest.json"
            manifest_path.write_text('{"status": "failed"}\n')

            archived = prepare_run_directory(
                run_directory, manifest_path, retry_failed=True
            )

            self.assertEqual(archived, run_directory / "manifest.attempt-1.json")
            self.assertFalse(manifest_path.exists())
            self.assertEqual(archived.read_text(), '{"status": "failed"}\n')

    def test_retry_refuses_non_failed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory) / "seed0"
            run_directory.mkdir()
            manifest_path = run_directory / "manifest.json"
            manifest_path.write_text('{"status": "complete"}\n')

            with self.assertRaisesRegex(FileExistsError, "status 'complete'"):
                prepare_run_directory(run_directory, manifest_path, retry_failed=True)

            self.assertTrue(manifest_path.exists())


if __name__ == "__main__":
    unittest.main()

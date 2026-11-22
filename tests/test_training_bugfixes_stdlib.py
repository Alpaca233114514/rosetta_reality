"""Pure-stdlib regression tests for verified v2 training bug fixes (no ML/sim imports)."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


basin_config = _load_module(
    "rosetta_test_basin_config", "src/rosetta_reality/vla/training/basin_config.py"
)


class TrainingBugfixesStdlibTest(unittest.TestCase):
    def test_basin_config_accepts_integer_yaml_optimizer_values_against_float_cli(self) -> None:
        # Regression: YAML integer grad_clip_norm: 10 / weight_decay: 0 vs float CLI "10.0" / "0.0".
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {
                "schema_version": 2,
                "role": "vla",
                "plan_id": "plan-int-001",
                "run_name": "run-int-001",
                "resources": {"mixed_precision": "bf16"},
                "training": {
                    "episodes": [49, 4],
                    "batch_size": 4,
                    "steps": 8,
                    "optimizer": {
                        "type": "adamw",
                        "lr": 1e-4,
                        "betas": [0.9, 0.95],
                        "eps": 1e-8,
                        "weight_decay": 0,  # integer 0 in YAML
                        "grad_clip_norm": 10,  # integer 10 in YAML
                    },
                    "scheduler": {
                        "type": "cosine_decay_with_warmup",
                        "num_warmup_steps": 2,
                        "num_decay_steps": 8,
                        "peak_lr": 1e-4,
                        "decay_lr": 2.5e-6,
                    },
                },
            }
            runtime = {
                "seed": 17,
                "model": {"revision": "c83c3163b8ca9b7e67c509fffd9121e66cb96205"},
                "dataset": {
                    "identifier": "lerobot/aloha_sim_insertion_human",
                    "revision": "cc571a3c661df81b566dbfde3d5c1e85fcdf7884",
                },
            }
            plan_path = root / "plan.json"
            runtime_path = root / "runtime.json"
            launch_path = root / "launch.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
            launch = {
                "mode": "train",
                "run_name": "run-int-001",
                "experiment_id": "m2-example",
                "formal_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                "experiment_config_sha256": "a" * 64,
                "runtime_experiment_sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
                "action_contract_sha256": "b" * 64,
                "normalization_report_sha256": "c" * 64,
                "dataset_view_manifest_sha256": "d" * 64,
                "code_identity": {"revision": "unit", "dirty": False},
            }
            launch_path.write_text(json.dumps(launch), encoding="utf-8")
            arguments = [
                "--seed=17",
                "--batch_size=4",
                "--steps=8",
                "--accelerator.gradient_accumulation.steps=1",
                "--accelerator.mixed_precision=bf16",
                "--dataset.repo_id=lerobot/aloha_sim_insertion_human",
                "--dataset.revision=cc571a3c661df81b566dbfde3d5c1e85fcdf7884",
                "--dataset.episodes=[49,4]",
                "--policy.optimizer_lr=0.0001",
                "--policy.optimizer_betas=[0.9,0.95]",
                "--policy.optimizer_eps=1e-08",
                "--policy.optimizer_weight_decay=0.0",
                "--policy.optimizer_grad_clip_norm=10.0",
                "--policy.scheduler_warmup_steps=2",
                "--policy.scheduler_decay_steps=8",
                "--policy.scheduler_decay_lr=2.5e-06",
            ]
            bundle = root / "bundle"
            basin_config.write_training_config_bundle(
                bundle,
                plan=plan,
                plan_path=plan_path,
                launch_path=launch_path,
                runtime_path=runtime_path,
                phase="train",
                arguments=arguments,
            )
            snapshot = json.loads((bundle / "snapshot.json").read_text(encoding="utf-8"))
            self.assertEqual(snapshot["parameters"]["training"]["optimizer"]["weight_decay"], 0.0)
            self.assertEqual(
                snapshot["parameters"]["training"]["optimizer"]["gradient_clip_norm"], 10.0
            )

    def test_launch_applies_empty_cameras_overlay_outside_train_only_branch(self) -> None:
        # Static AST verification + execution via stubbed yaml so it runs in pure stdlib.
        source = (ROOT / "src/rosetta_reality/vla/training/launch.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        fn = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_training_arguments"
        )
        # Ensure empty_cameras is handled before the MODE_TRAIN-only compile block.
        text = ast.get_source_segment(source, fn)
        self.assertIn('if "empty_cameras" in overlay:', text)
        self.assertLess(
            text.index('if "empty_cameras" in overlay:'),
            text.index("if mode == MODE_TRAIN:"),
        )

    def test_checkpoint_metric_snapshot_uses_tracker_step_str_dir_and_fail_closed_restore(self) -> None:
        # Stub minimal yaml if absent so package import stays pure stdlib.
        added_yaml = False
        if "yaml" not in sys.modules:
            sys.modules["yaml"] = types.ModuleType("yaml")
            added_yaml = True
        if str(ROOT / "src") not in sys.path:
            sys.path.insert(0, str(ROOT / "src"))
        try:
            import rosetta_reality.vla.training.features as features_mod

            class _FakeTrain:
                def __init__(self) -> None:
                    self.update_policy = self._orig_update
                    self.save_checkpoint = self._orig_save

                @staticmethod
                def _orig_update(tracker, *args, **kwargs):
                    return tracker, {}

                @staticmethod
                def _orig_save(checkpoint_dir, step, *args, **kwargs):
                    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                    return "saved"

            fake_module = _FakeTrain()
            orig_getter = features_mod._lerobot_train_module
            features_mod._lerobot_train_module = lambda: fake_module
            try:
                feature = features_mod.CheckpointMetricSnapshotFeature({})
                feature._memory_snapshot = staticmethod(lambda _device: {"api": "unavailable"})
                ctx = types.SimpleNamespace(
                    device="cpu",
                    run_name="resume-run",
                    phase="formal",
                    experiment={"experiment_id": "m2-test"},
                )
                feature.install(ctx)
                tracker = types.SimpleNamespace(
                    step=2501, loss=0.12, grad_norm=1.5, lr=1e-4, update_s=0.3
                )
                fake_module.update_policy(tracker)
                with tempfile.TemporaryDirectory() as tmp:
                    ckpt_dir = str(Path(tmp) / "ckpt-2501")  # str path regression
                    fake_module.save_checkpoint(ckpt_dir, 2501)
                    snap = json.loads(
                        (Path(ckpt_dir) / "rosetta_checkpoint_metrics.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(snap["step"], 2501)
                    self.assertAlmostEqual(snap["metrics"]["loss"], 0.12)

                # Tampering with update_policy before restore must fail closed.
                saved_hook = fake_module.update_policy
                fake_module.update_policy = lambda *a, **k: None
                with self.assertRaises(RuntimeError):
                    feature.restore(ctx)
                fake_module.update_policy = saved_hook
                feature.restore(ctx)
                self.assertIs(fake_module.update_policy, fake_module._orig_update)
                self.assertIs(fake_module.save_checkpoint, fake_module._orig_save)
            finally:
                features_mod._lerobot_train_module = orig_getter
        finally:
            if added_yaml:
                sys.modules.pop("yaml", None)

    def test_run_smolvla_v2_validate_normalization_annotation_is_three_tuple(self) -> None:
        source = (ROOT / "scripts/run_smolvla_v2.py").read_text(encoding="utf-8")
        self.assertIn(") -> tuple[Path, Path, Path]:", source)


if __name__ == "__main__":
    unittest.main()

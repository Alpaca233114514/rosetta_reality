"""File-only registration preparation and historical evidence indexing. Never launches ML."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rosetta_reality.diagnostics.dual_axis import (
    EvidenceWriter,
    encoded,
    make_panels,
    make_schedule,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "faust": "m2-smolvla-faust-trainer-optimizer-audit-2026-08-12.md",
    "zen": "m2-smolvla-zen-formal-audit-2026-08-27.md",
    "vcdropout": "m2-smolvla-vcdropout-gate34-result-2026-08-29.md",
    "vfunfreeze": "m2-smolvla-vfunfreeze-gate34-closure-2026-09-06.md",
    "hestia": "m2-smolvla-hestia-main-recovery-result-2026-09-11.md",
    "iris": "m2-smolvla-iris-result-002-2026-09-13.md",
    "canonical": "m2-smolvla-canonical-fullframes-gate34-result-2026-09-15.md",
    "canonical-curve": "m2-smolvla-visual-process-result-2026-09-16.md",
    "first-step": "m2-smolvla-gate-first-step-review-2026-09-16.md",
    "grasp-loss": "m2-smolvla-gate-grasp-loss-review-2026-09-16.md",
    "prepost": "m2-smolvla-prepost-analysis-and-next-plan-2026-09-24.md",
}
BRANCHES = (
    ("Faust", None, "fresh_base", "bounded action boundary", "faust"),
    (
        "Aster",
        "Faust",
        "fresh_base",
        "first-action weighting; distinguish defective -002 and corrected -003",
        "zen",
    ),
    (
        "Way",
        "Aster",
        "fresh_base",
        "state jitter; batch/update/runtime changes are confounds",
        "zen",
    ),
    ("Zen-uniform", "Way", "fresh_base", "matched uniform control", "zen"),
    ("Zen-firstaction", "Zen-uniform", "fresh_base", "matched first-action weighting", "zen"),
    ("vcdropout", "Zen-uniform", "fresh_base", "state conditioning dropout", "vcdropout"),
    ("vfunfreeze", "vcdropout", "fresh_base", "vision-front-end adaptation", "vfunfreeze"),
    ("Hestia", "Zen-uniform", "fresh_base", "frame-0 coverage/fit-budget investigation", "hestia"),
    ("Iris-control", "Hestia", "fresh_base", "registered control replication", "iris"),
    ("Iris-treatment", "Iris-control", "fresh_base", "image K scene regularization", "iris"),
    (
        "canonical",
        "Hestia",
        "fresh_base",
        "canonical image processing and full-frame coverage",
        "canonical",
    ),
    (
        "prepost",
        "canonical",
        "saved_endpoints",
        "base/2500/5000 and separate Gate endpoint comparison",
        "prepost",
    ),
)


def historical_index(output):
    writer = EvidenceWriter(
        output,
        {
            "source_run": "dual-axis-historical-index-20260924",
            "evidence_kind": "historical_observation",
        },
        maximum_bytes=4 * 1024**2,
    )
    refs = {}
    facts = {}
    for key, name in SOURCES.items():
        path = ROOT / "reports/training" / name
        raw = path.read_bytes()
        refs[key] = {
            "repository_path": path.relative_to(ROOT).as_posix(),
            **writer.write(f"sources/{name}", raw, raw=True),
        }
        companion = path.with_suffix(".json")
        facts[key] = []
        if companion.is_file():
            companion_raw = companion.read_bytes()
            refs[key]["machine_source"] = writer.write(
                f"sources/{companion.name}", companion_raw, raw=True
            )

            def walk(value, pointer=""):
                if isinstance(value, dict):
                    for field, item in value.items():
                        current = pointer + "/" + field.replace("~", "~0").replace("/", "~1")
                        if any(
                            word in field
                            for word in (
                                "checkpoint",
                                "batch_size",
                                "optimizer_steps",
                                "exposures",
                                "runtime",
                                "episode",
                                "selected_step",
                            )
                        ):
                            if len(encoded(item)) <= 4096:
                                facts[key].append({"pointer": current, "value": item})
                        walk(item, current)
                elif isinstance(value, list):
                    for index, item in enumerate(value):
                        walk(item, pointer + f"/{index}")

            walk(json.loads(companion_raw))
    for name, parent, inheritance, axis, source in BRANCHES:
        writer.event(
            {"axis": "history", "branch": name},
            {
                "research_parent": parent,
                "weight_ancestry": inheritance,
                "changed_axis": axis,
                "source": refs[source],
                "source_facts": facts[source][:64],
                "source_fact_count": len(facts[source]),
                "research_relation_status": "curated_navigation_not_causal_or_weight_lineage",
                "data_and_runtime": "source-scoped facts; not assumed comparable",
                "missing": ["dense_per_update_gradients", "dense_fixed_input_activations"],
                "historical_reconstruction": False,
            },
            reasons=["history_index"],
        )
    for source, interval, meaning in (
        (
            "hestia",
            [640, 1280],
            "Late local regression; does not explain the pre-existing generalization gap",
        ),
        (
            "canonical-curve",
            [2500, 5000],
            "Frame125 local development regression; other groups can improve",
        ),
        ("first-step", [0, 0], "First-action deviation is not a proven irreversible failure cause"),
        (
            "grasp-loss",
            [122, 127],
            "Old trace grasp retention window; not transferable to new runs",
        ),
    ):
        writer.event(
            {"axis": "history", "branch": source},
            {
                "priority_window": interval,
                "axis": "execution" if source in ("first-step", "grasp-loss") else "training",
                "interpretation": meaning,
                "source": refs[source],
            },
            reasons=["historical_branch_point"],
        )
    return writer.seal()


def prepare(output, *, total, train, dev, hidden, warmup=0, checkpoints=(), epochs=(), branches=()):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    noises = [{"name": "zero", "seed": None}] + [
        {"name": f"seed_{s}", "seed": s} for s in (20260905, 20260906, 20260907)
    ]
    plan = {
        "schema_version": 1,
        "status": "draft",
        "source_run": "dual-axis-short-diagnostic-draft",
        "model_execution_authorized": False,
        "training_authorized": False,
        "rollout_authorized": False,
        "total_updates": total,
        "schedule": make_schedule(
            total,
            warmup=warmup,
            epoch_boundaries=epochs,
            checkpoints=checkpoints,
            branches=branches,
        ),
        "panels": make_panels(train, dev, hidden, noises),
        "modules": [],
        "ring_bytes": 256 * 1024**2,
        "maximum_bytes": 4 * 1024**3,
        "maximum_routine_overhead": 0.15,
        "host_memory_budget_gib": 16,
        "probe_bank": {},
        "action_groups": {},
        "probe_bank_root": None,
        "admission": None,
        "missing_admission": [
            "sealed real-input probe bank",
            "exact module inventory and token route identity",
            "registered_model AA/light/full parity",
            "measured routine and anomaly costs",
            "current runtime/source identity",
            "explicit compute deadline and budget",
        ],
        "endpoint_gradient_stage": {"split": "train", "optimizer_updates": 0},
        "gate_status": "unchanged",
    }
    members = [
        ROOT / "scripts/prepare_dual_axis.py",
        *sorted((ROOT / "src/rosetta_reality/diagnostics").glob("dual_axis*.py")),
        ROOT / "src/rosetta_reality/vla/training/dual_axis.py",
        ROOT / "src/rosetta_reality/vla/training/features.py",
        ROOT / "src/rosetta_reality/vla/training/plan.py",
        ROOT / "src/rosetta_reality/vla/training/integrity.py",
        ROOT / "src/rosetta_reality/vla/training/launch.py",
        ROOT / "scripts/run_smolvla_v2.py",
        ROOT / "scripts/train_smolvla_v2.py",
        ROOT / "src/rosetta_reality/eval/dual_axis.py",
        ROOT / "src/rosetta_reality/eval/reproducibility_capture.py",
    ]
    plan["implementation_files"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in members
    }
    (output / "plan.json").write_bytes(encoded(plan))
    historical_index(output / "history")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument(
        "--split", type=Path, required=True, help="JSON containing train/dev/hidden episode lists"
    )
    args = parser.parse_args()
    split = json.loads(args.split.read_text())
    prepare(args.output, total=args.total, warmup=args.warmup, **split)


if __name__ == "__main__":
    main()

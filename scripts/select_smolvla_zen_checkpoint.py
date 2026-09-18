"""Select one Zen checkpoint per registered protocol, then compare both arms.

Selection is validation-only and preregistered: the primary metric is
``first_action_mae`` from the fixed-validation reports produced by
``smolvla_zen_validate``; ties break toward the earlier checkpoint. The base
model report participates as candidate ``base`` but never wins by
construction unless every checkpoint regressed beyond it (recorded honestly).

Usage:
    python scripts/select_smolvla_zen_checkpoint.py --plan <zen-plan> \
        --run-root <ROSETTA_RUN_ROOT>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (str(REPOSITORY_ROOT / "src"), SCRIPTS_ROOT):
    if root not in sys.path:
        sys.path.insert(0, root)

import smolvla_zen_protocol as protocol  # noqa: E402

from rosetta_reality.experiment import file_sha256, workspace_code_identity  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402

PRIMARY_METRIC = "first_action_mae"
SECONDARY_METRIC = "fixed_flow_loss"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    plan_path = args.plan.resolve()
    plan, plan_id = protocol.resolve_plan(plan_path)
    spec = protocol.ZEN_SPECS[plan_id]
    prefix = spec["validation_prefix"]
    validation_root = args.run_root.resolve() / protocol.EXPERIMENT_ID / "validation"

    candidates: list[dict] = []
    base_label = f"{prefix}-base.json"
    labels = [base_label] + [
        f"{prefix}-step-{step:06d}.json" for step in protocol.CHECKPOINT_STEPS
    ]
    for label in labels:
        path = validation_root / label
        if not path.is_file():
            raise FileNotFoundError(f"Missing validation report: {label}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "complete":
            raise ValueError(f"Validation report not complete: {label}")
        metrics = report["metrics"]
        value = float(metrics[PRIMARY_METRIC])
        secondary = float(metrics[SECONDARY_METRIC])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite primary metric in {label}")
        source = report["model_source"]
        candidates.append(
            {
                "label": label,
                "kind": source.get("kind"),
                "checkpoint_step": source.get("checkpoint_step", source.get("step")),
                PRIMARY_METRIC: value,
                SECONDARY_METRIC: secondary,
                "report_sha256": file_sha256(path),
                "action_mae": float(metrics["action_mae"]),
                "invalid_action_rate": float(metrics["invalid_action_rate"]),
                "joint_limit_violation_rate": float(
                    metrics["joint_limit_violation_rate"]
                ),
            }
        )

    checkpoint_rows = [row for row in candidates if row["kind"] == "checkpoint"]
    best = min(
        checkpoint_rows,
        key=lambda row: (
            row[PRIMARY_METRIC],
            -int(row["checkpoint_step"] or 0),
        ),
    )
    base_row = next(row for row in candidates if row["kind"] == "base")
    selected_step = int(best["checkpoint_step"])

    decision = {
        "schema_version": 1,
        "status": "selected",
        "stage": "smolvla_zen_validation_only_selection",
        "experiment_id": protocol.EXPERIMENT_ID,
        "plan_id": plan_id,
        "plan_sha256": file_sha256(plan_path),
        "role": spec["role"],
        "primary_metric": PRIMARY_METRIC,
        "tie_break": "earlier_checkpoint",
        "candidates": sorted(candidates, key=lambda r: r[PRIMARY_METRIC]),
        "selected_checkpoint_step": selected_step,
        "selected_primary_value": best[PRIMARY_METRIC],
        "base_primary_value": base_row[PRIMARY_METRIC],
        "improvement_over_base_fraction": (
            (base_row[PRIMARY_METRIC] - best[PRIMARY_METRIC]) / base_row[PRIMARY_METRIC]
        ),
        "read_only_context_controls": {
            "note": (
                "Offline values recorded as context only; they originate from "
                "different training identities and are not same-tree comparisons."
            ),
            "aster_offline_control_first_action_mae": 0.02250973408226855,
            "prometheus_early_horizon_first_action_mae": 0.02914831822854467,
        },
        "hidden_test_loaded": False,
        "zen_protocol": {
            "wrapper_sha256": file_sha256(Path(__file__)),
            "protocol_module_sha256": file_sha256(
                REPOSITORY_ROOT / "scripts/smolvla_zen_protocol.py"
            ),
            "code_identity": workspace_code_identity(REPOSITORY_ROOT),
        },
    }
    destination = (
        args.run_root.resolve()
        / protocol.EXPERIMENT_ID
        / "selection"
        / f"{spec['run_name']}-selection.json"
    )
    create_json(destination, decision)
    print(json.dumps({"selection": str(destination), "selected_step": selected_step}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

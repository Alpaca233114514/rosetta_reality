"""Select the Aster checkpoint while preserving the repaired action boundary."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import run_smolvla_horizon_loss_formal as formal_runner  # noqa: E402
import select_smolvla_checkpoint as selector  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.vla import load_smolvla_action_space  # noqa: E402


def _apply_control_acceptance(
    payload: dict[str, Any],
    plan: dict[str, Any],
    control_plan: dict[str, Any],
    control_report: dict[str, Any],
) -> bool:
    """Require the registered Aster metric to improve over the Faust control."""

    control = plan["control_reference"]
    primary = str(plan["validation"]["primary_selection_metric"])
    candidate_metrics = payload.get("selected", {}).get("metrics", {})
    control_metrics = control_report.get("selected", {}).get("metrics", {})
    candidate = candidate_metrics.get(primary)
    baseline = control_metrics.get(primary)
    if (
        primary != "first_action_mae"
        or control_plan.get("run_name") != control.get("control_run")
        or control_report.get("status") != "passed"
        or control_report.get("stage") != "smolvla_formal_checkpoint_selection"
        or control_report.get("experiment_id") != payload.get("experiment_id")
        or control_report.get("formal_plan_sha256") != control.get("plan_sha256")
        or isinstance(candidate, bool)
        or not isinstance(candidate, int | float)
        or not math.isfinite(float(candidate))
        or isinstance(baseline, bool)
        or not isinstance(baseline, int | float)
        or not math.isfinite(float(baseline))
    ):
        raise ValueError("The registered Faust control comparison is invalid.")
    improves = float(candidate) < float(baseline)
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict) or not acceptance:
        raise ValueError("The Aster selection payload has no acceptance evidence.")
    acceptance["validation_first_action_mae_improves_over_faust_control"] = improves
    payload["control_comparison"] = {
        "control_run": control["control_run"],
        "selection_report_sha256": control["selection_report_sha256"],
        "metric": primary,
        "control_value": float(baseline),
        "candidate_value": float(candidate),
        "absolute_improvement": float(baseline) - float(candidate),
        "relative_improvement": (float(baseline) - float(candidate)) / float(baseline),
    }
    passed = all(value is True for value in acceptance.values())
    payload["status"] = "passed" if passed else "rejected"
    return passed


def _mark_public_sync_complete(
    payload: dict[str, Any], plan: dict[str, Any]
) -> None:
    """Record the public sync already validated by the delegated selector."""

    if (
        not isinstance(payload.get("trackio_sync_report_sha256"), str)
        or len(payload["trackio_sync_report_sha256"]) != 64
        or not isinstance(payload.get("trackio_project_snapshot_sha256"), str)
        or len(payload["trackio_project_snapshot_sha256"]) != 64
        or not isinstance(payload.get("trackio_synced_run"), dict)
    ):
        raise ValueError("Aster selection has no validated public sync provenance.")
    payload.update(
        {
            "trackio_delivery_status": "public_checkpoint_sync_complete",
            "public_sync_performed": True,
            "bounded_gripper_decoder": True,
            "temporal_loss_profile": plan["loss_contract"]["profile"],
            "temporal_loss_normalization": plan["loss_contract"][
                "normalization"
            ],
            "selection_script_sha256": file_sha256(Path(__file__)),
        }
    )


def main() -> int:
    plan_path = (
        Path(sys.argv[sys.argv.index("--plan") + 1]).resolve()
        if "--plan" in sys.argv
        else formal_runner.DEFAULT_PLAN.resolve()
    )
    plan, _base_path, _experiment = formal_runner._validate_plan(plan_path)
    control = plan["control_reference"]
    control_plan_path = formal_runner._repository_path(str(control["plan"]))
    control_report_path = formal_runner._repository_path(
        str(control["selection_report"])
    )
    if (
        file_sha256(control_plan_path) != control["plan_sha256"]
        or file_sha256(control_report_path) != control["selection_report_sha256"]
    ):
        raise ValueError("The registered Faust control files changed.")
    control_plan = formal_runner._load_yaml(control_plan_path)
    control_report = formal_runner._load_json(control_report_path)
    selector.formal_runner = formal_runner
    original_validation = selector._validation_report
    original_create_json = selector.create_json
    aster_passed: list[bool] = []

    def validation_report(path: Path, **kwargs: Any) -> dict[str, Any]:
        report = original_validation(path, **kwargs)
        experiment = kwargs["experiment"]
        if (
            report.get("action_space")
            != load_smolvla_action_space(experiment, require_explicit=True).as_dict()
            or report.get("bounded_gripper_decoder") is not True
        ):
            raise ValueError("An Aster validation report lost the repaired action boundary.")
        return report

    def create_json(path: Path, payload: dict[str, Any]) -> None:
        passed = _apply_control_acceptance(
            payload, plan, control_plan, control_report
        )
        _mark_public_sync_complete(payload, plan)
        original_create_json(path, payload)
        aster_passed.append(passed)

    selector._validation_report = validation_report
    selector.create_json = create_json
    result = selector.main()
    if len(aster_passed) != 1:
        raise RuntimeError("Aster selection did not create exactly one report.")
    if result != 0:
        return result
    return 0 if aster_passed[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())

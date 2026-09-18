"""Select from sealed validation reports under a new v2 plan; never rewrite old selections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from run_smolvla_v2 import _resolve_plan, _validate_split  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.vla.checkpoint_selection import select_reports  # noqa: E402
from rosetta_reality.vla.training.integrity import validate_local_implementation  # noqa: E402
from rosetta_reality.vla.training.plan import repository_relative_path  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Selection output is create-only")
    plan, parent, experiment = _resolve_plan(args.plan.resolve())
    verified = validate_local_implementation(plan, ROOT)
    if (
        not {
            "scripts/select_smolvla_checkpoint_v2.py",
            "src/rosetta_reality/vla/checkpoint_selection.py",
        }
        <= verified.keys()
    ):
        raise ValueError("New selection implementation must be sealed by the plan")
    _validate_split(plan, experiment, "train")
    contract = plan.get("selection_contract", {})
    if (
        contract.get("primary_metric") != "first_action_mae"
        or contract.get("tie_break") != "earlier_checkpoint"
        or contract.get("base_role") != "reference_only"
        or contract.get("fixed_input") != {"noise": "zeros", "flow_time": 0.5}
    ):
        raise ValueError("Selection requires the registered v2 metric/tie/base/noise contract")
    if file_sha256(args.inputs) != args.inputs_sha256:
        raise ValueError("Selection input receipt checksum changed")
    inputs = json.loads(args.inputs.read_text())
    plan_sha = file_sha256(args.plan)
    if inputs.get("plan_sha256") != plan_sha:
        raise ValueError("Selection inputs belong to another plan")
    records = []
    for entry in inputs["reports"]:
        relative = repository_relative_path(entry["path"], context="Validation report")
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or file_sha256(path) != entry["sha256"]:
            raise ValueError("Validation report checksum or path differs")
        records.append(
            {
                "report": json.loads(path.read_text()),
                "report_sha256": entry["sha256"],
                "source": entry["source"],
            }
        )
    identity = {
        "formal_plan_sha256": plan_sha,
        "experiment_id": experiment["experiment_id"],
        "experiment_config_sha256": file_sha256(parent),
        "action_contract_sha256": file_sha256(ROOT / experiment["action_contract"]["derived"]),
        "normalization_report_sha256": plan["normalization"]["report_sha256"],
        "dataset_view_manifest_sha256": plan["normalization"]["dataset_view_manifest_sha256"],
        "model_revision": experiment["model"]["revision"],
        "dataset_revision": experiment["dataset"]["revision"],
    }
    result = select_reports(
        records,
        identity=identity,
        validation=plan["validation"],
        checkpoint_steps=plan["training"]["checkpoint_steps"],
        fixed_input=contract["fixed_input"],
    )
    result.update(
        plan_id=plan["plan_id"], inputs_sha256=args.inputs_sha256, implementation_sha256=verified
    )
    create_json(args.output, result)
    print(json.dumps({"status": result["status"], "step": result["selected_checkpoint_step"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

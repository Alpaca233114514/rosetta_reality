"""Select the Faust checkpoint while preserving the repaired action boundary."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import run_smolvla_action_repair_formal as formal_runner  # noqa: E402
import select_smolvla_checkpoint as selector  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.vla import load_smolvla_action_space  # noqa: E402


def main() -> int:
    selector.formal_runner = formal_runner
    original_validation = selector._validation_report
    original_create_json = selector.create_json

    def validation_report(path: Path, **kwargs: Any) -> dict[str, Any]:
        report = original_validation(path, **kwargs)
        experiment = kwargs["experiment"]
        if (
            report.get("action_space")
            != load_smolvla_action_space(experiment, require_explicit=True).as_dict()
            or report.get("bounded_gripper_decoder") is not True
        ):
            raise ValueError("A Faust validation report lost the repaired action boundary.")
        return report

    def create_json(path: Path, payload: dict[str, Any]) -> None:
        readiness = payload.pop("trackio_sync_report_sha256")
        payload.update(
            {
                "trackio_space_readiness_report_sha256": readiness,
                "trackio_delivery_status": "local_durable_pending_public_sync",
                "public_sync_performed": False,
                "bounded_gripper_decoder": True,
                "selection_script_sha256": file_sha256(Path(__file__)),
            }
        )
        original_create_json(path, payload)

    selector._validation_report = validation_report
    selector.create_json = create_json
    return selector.main()


if __name__ == "__main__":
    raise SystemExit(main())

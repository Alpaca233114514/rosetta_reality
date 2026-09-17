"""Create-only full-array proof; do not promote equal scalar metrics to tensor equality."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.vla.reload_evidence import compare_bundles  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--identity-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Reload proof is create-only")
    if file_sha256(args.identity) != args.identity_sha256:
        raise ValueError("Expected reload identity checksum drift")
    result = compare_bundles(
        args.first, args.second, expected_identity=json.loads(args.identity.read_text())
    )
    result["identity_file_sha256"] = args.identity_sha256
    create_json(args.output, result)
    print(
        json.dumps(
            {"status": result["status"], "exact_tensor_equality": result["exact_tensor_equality"]}
        )
    )
    return 0 if result["exact_tensor_equality"] else 4


if __name__ == "__main__":
    raise SystemExit(main())

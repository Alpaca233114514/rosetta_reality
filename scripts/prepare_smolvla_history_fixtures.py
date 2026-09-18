"""Extract only stale historical implementation bytes from local Git for tests."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PLANS = (
    "smolvla_450m_aloha_insertion_zen_cuda_b64_uniform_002.yaml",
    "smolvla_450m_aloha_insertion_zen_cuda_b64_firstaction_001.yaml",
    "smolvla_450m_aloha_insertion_vfunfreeze_cuda_b32_003.yaml",
)
VISION_SNAPSHOT_SHA = "0ce44efe4cf02d537e1b225daed3b039c604c6c85a66f8993bfb749f56e1797b"
SNAPSHOTS = {
    VISION_SNAPSHOT_SHA: "runs/smolvla-visual-repair-20260909-001/before/vision_front_end.py",
}


def git(*args):
    return subprocess.check_output(["git", "-c", f"safe.directory={ROOT}", *args], cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plans", nargs="+", default=PLANS)
    args = parser.parse_args()
    expected = {}
    for name in args.plans:
        if Path(name).name != name:
            raise ValueError("Plan names must be files under configs/vla")
        plan = yaml.safe_load((ROOT / "configs/vla" / name).read_text())
        expected.update({(path, sha): None for path, sha in plan["implementation_files"].items()})
    records, blobs, missing = {}, {}, []
    for name, sha in sorted(expected):
        current = ROOT / name
        if current.is_file() and hashlib.sha256(current.read_bytes()).hexdigest() == sha:
            continue
        snapshot = ROOT / SNAPSHOTS[sha] if sha in SNAPSHOTS else None
        if snapshot is not None and snapshot.is_file():
            content = snapshot.read_bytes()
            if hashlib.sha256(content).hexdigest() != sha:
                raise ValueError("Historical snapshot checksum drift")
            records[sha] = {
                "source_path": name,
                "evidence_source": SNAPSHOTS[sha],
                "bytes": len(content),
            }
            blobs[sha] = content
            continue
        revisions = git("log", "--all", "--format=%H", "--", name).decode().splitlines()
        for revision in revisions:
            try:
                content = git("show", f"{revision}:{name}")
            except subprocess.CalledProcessError:
                continue
            variants = (content, content.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
            content = next((b for b in variants if hashlib.sha256(b).hexdigest() == sha), None)
            if content is not None:
                records[sha] = {
                    "source_path": name,
                    "git_revision": revision,
                    "git_blob": git("rev-parse", f"{revision}:{name}").decode().strip(),
                    "bytes": len(content),
                }
                blobs[sha] = content
                break
        else:
            missing.append({"source_path": name, "sha256": sha})
    args.output.mkdir(parents=True, exist_ok=False)
    for sha, content in blobs.items():
        (args.output / f"{sha}.txt").write_bytes(content)
    (args.output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(
        json.dumps(
            {
                "fixtures": len(records),
                "bytes": sum(len(b) for b in blobs.values()),
                "missing": missing,
            }
        )
    )
    if missing:
        raise RuntimeError("Exact historical source unavailable; partial fixtures retained")


if __name__ == "__main__":
    main()

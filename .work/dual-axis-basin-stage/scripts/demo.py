"""Synthetic demo plus optional read-only import from a neighboring Rosetta repo."""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basin.adapters import import_gate, import_native, import_trace
from basin.analysis import analyze, compare
from basin.io import child, dumps, outside, read_bytes, sha, write_new
from basin.report import render
from basin.store import Store


def source_snapshot(repo, sources):
    """Only named input files plus tracked changes; no scan of weights or datasets."""
    def git(*args):
        return subprocess.check_output(["git", "--no-optional-locks", "-C", str(repo), *args])
    return {"head": git("rev-parse", "HEAD").decode().strip(),
            "status_sha256": sha(git("status", "--porcelain=v1", "--untracked-files=all")),
            "tracked_diff_sha256": sha(git("diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD")),
            "inputs": {label: sha(read_bytes(child(root, name))) for label, root, name in sources}}


def run(output, rosetta=None, trace_root=None, html=False):
    output = Path(output).resolve()
    if rosetta:
        rosetta = Path(rosetta).resolve()
        outside(output, rosetta)
    if trace_root:
        if not rosetta:
            raise ValueError("--trace-root requires --rosetta")
        trace_root = Path(trace_root).resolve()
        outside(output, trace_root)
    output.mkdir(parents=True, exist_ok=False)
    store = Store(output / "history")
    for label in ("control", "changed"):
        envelope = {"schema_version": 1, "evidence_kind": "synthetic", "status": "complete",
                    "source_run": "toy-" + label, "dimensions": ["toy_action"],
                    "parameters": {"seed": 0, "condition": label}, "events": []}
        for step in range(8):
            delta = 0.5 if label == "changed" and step >= 3 else 0.0
            envelope["events"].append({"step": step, "stage": "execution", "values": {
                "action": [step / 10 + delta], "reward": 0.0, "success": False,
                "loss": 1 / (step + 1) + delta, "gradient_norm": 0.2 + delta,
                "activation": [0.1, step / 20 + delta]}})
        source = output / ("toy-" + label + ".json")
        write_new(source, dumps(envelope))
        import_native(store, source, "toy-" + label)
    toy_diff = compare(store, "toy-control", "toy-changed")
    write_new(output / "synthetic-comparison.json", dumps(toy_diff))
    receipt = {"synthetic_first_difference": toy_diff["first_recorded_difference"]["step"],
               "real_model_executed": False, "rosetta_integration": "not_requested"}
    displayed = toy_diff
    if rosetta:
        base = "traces" if trace_root else "runs/canonical-posttrain-received-20260916-005-ssh/verified/traces"
        evidence_root = trace_root or rosetta
        report = "reports/training/m2-smolvla-traced-gate-result-2026-09-16.json"
        traces = [(f"{base}/gate4-{seed}", f"rosetta-gate4-{seed}") for seed in range(1000, 1005)]
        sources = [(report, rosetta, report)] + [(f"evidence/{path}/{name}", evidence_root, f"{path}/{name}")
                   for path, _ in traces for name in ("identity.json", "summary.json", "trace.jsonl", "manifest.json")]
        before = source_snapshot(rosetta, sources)
        write_new(output / "rosetta-before.json", dumps(before))
        import_gate(store, rosetta, report, "rosetta-gate-report")
        for path, run_id in traces:
            import_trace(store, evidence_root, path, run_id, gate="gate4")
            write_new(output / (run_id + "-analysis.json"), dumps(analyze(store, run_id)))
        displayed = compare(store, "rosetta-gate4-1000", "rosetta-gate4-1002")
        write_new(output / "rosetta-comparison.json", dumps(displayed))
        after = source_snapshot(rosetta, sources)
        write_new(output / "rosetta-after.json", dumps(after))
        receipt.update({"rosetta_integration": "existing_trace_imported",
                        "named_source_files": len(sources), "source_snapshot_unchanged": before == after,
                        "source_snapshot_scope": "HEAD, Git status, tracked diff, and named input bytes",
                        "completed_steps": sum(analyze(store, rid)["facts"]["completed_execution_steps"]
                                               for _, rid in traces)})
        if before != after:
            write_new(output / "receipt.json", dumps(receipt))
            raise ValueError("Rosetta snapshot changed during import; inspect before/after evidence")
    if html:
        render(store, output / "index.html", displayed)
    write_new(output / "receipt.json", dumps(receipt))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/demo")
    parser.add_argument("--rosetta", help="Optional read-only Rosetta checkout")
    parser.add_argument("--trace-root", help="Explicit read-only evidence root for offloaded runs; contains traces/")
    parser.add_argument("--html", action="store_true", help="Optional human-facing HTML export; not needed by the API")
    args = parser.parse_args()
    print(dumps(run(args.output, args.rosetta, args.trace_root, args.html)))

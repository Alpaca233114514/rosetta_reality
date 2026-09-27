"""Compact readout of verified diagnostics; no policy or simulator execution."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from analyze_root_neighborhoods import ROOT, seal, sha, write
from prepare_root_diagnosis import GATE_SUBDIR, GATES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = ROOT / "runs/root-neighborhoods-20260922-002"
    verification = ROOT / "runs/root-neighborhoods-independent-20260922-002/result.json"
    receipt = json.loads(verification.read_text())
    if receipt["status"] != "passed" or receipt["input_manifest_sha256"] != sha(
        source / "manifest.json"
    ):
        raise ValueError("Missing matching independent verification")
    groups = defaultdict(list)
    with (source / "metrics.csv").open() as stream:
        for row in csv.DictReader(stream):
            keys = ("frame", "mode", "group", "window", "checkpoint", "noise", "split")
            key = tuple(row[k] for k in keys)
            groups[key].append(row)
    summaries = []
    for key, rows in groups.items():
        metric_names = (
            "model_mae",
            "model_mse",
            "neighbor_mae",
            "mean_mae",
            "median_mae",
            "neighbor_label_std",
        )
        summaries.append(
            {
                **dict(zip(keys, key, strict=True)),
                **{k: mean(float(r[k]) for r in rows) for k in metric_names},
                "episodes": len(rows),
                "model_better_than_neighbor_count": sum(
                    float(r["model_mae"]) < float(r["neighbor_mae"]) for r in rows
                ),
            }
        )
    cases = []
    for line in (source / "events.jsonl").read_text().splitlines():
        r = json.loads(line)
        if (
            r["episode"] in (13, 22, 33)
            and r["frame"] == 125
            and r["checkpoint"] == 5000
            and r["noise"] == 0
            and r["side"] == "left_gripper"
            and 0.52 < r["threshold"] < 0.54
        ):
            cases.append(r)
    gates = {}
    for attempt, directory in GATES.items():
        path = ROOT / directory / GATE_SUBDIR / "gate4-smolvla-sim-471.json"
        gate = json.loads(path.read_text())
        gates[attempt] = {
            "aggregate": gate["aggregate"],
            "status": gate["status"],
            "source": path.relative_to(ROOT).as_posix(),
            "sha256": sha(path),
        }
    inventory = json.loads(
        (ROOT / "runs/root-evidence-20260922-001/evidence-inventory.json").read_text()
    )
    counts = defaultdict(int)
    for r in inventory["files"]:
        counts[r["status"]] += 1
    args.output.mkdir(parents=True, exist_ok=False)
    write(
        args.output / "result.json",
        {
            "status": "completed_exploratory_readout",
            "summaries": summaries,
            "case_events": cases,
            "historical_gates": gates,
            "inventory_status_counts": dict(counts),
            "independent_receipt": receipt,
            "limitations": [
                "No policy execution; development has only five previously used episodes.",
                "Frame0 state neighbors tie; episode-ID ordering does not measure similarity.",
                "Runtime hooks have synthetic tests, not real trainer acceptance.",
            ],
        },
    )
    seal(args.output)
    print(
        json.dumps(
            {
                "cases": cases,
                "focus": [
                    r
                    for r in summaries
                    if r["split"] == "development"
                    and r["group"] == "left_gripper"
                    and r["checkpoint"] == "5000"
                    and r["window"] == "full"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

"""Persist a synthetic localization example plus actual CLI/MCP protocol evidence."""
import json
import subprocess
import sys
from pathlib import Path

from basin.api import BasinAPI
from basin.dual_axis import import_dual_axis
from basin.store import Store
from rosetta_reality.diagnostics.dual_axis import EvidenceWriter

output = Path("/output")
source = output / "inputs"
source.mkdir()
identity = {key: key + "-synthetic" for key in ("input", "noise", "processor", "action_contract", "runtime")}
for name, values in (("control", [(0, 1), (64, 1), (128, 1), (129, 1), (130, 1)]),
                     ("deviation", [(0, 1), (64, 1), (128, 1.5), (129, 1.6), (130, 1.05)])):
    writer = EvidenceWriter(source / name, {"source_run": "synthetic-dual-axis-demonstration",
                                           "evidence_kind": "synthetic", "schedule": [{"update": u} for u, _ in values],
                                           "lineage": {"research_parent": "synthetic-control", "weight_parent": None}},
                            maximum_bytes=1024**2)
    for update, error in values:
        writer.event({"axis": "probe", "update": update, "sample_id": "train-2-frame0", "episode": 2,
                      "frame": 0, "noise_id": "seed_20260905", "boundary": "model_output"},
                     {"metrics": [{"name": "left_joint/first_action/mae", "category": "behavior",
                                   "group": "left_joint", "window": "first_action", "value": error, "aa_floor": 0.01}]},
                     identity=identity, reasons=["regular"] if update in (0, 64, 128) else ["anomaly"])
    writer.seal()

store = Store(output / "basin-history")
for name in ("control", "deviation"):
    import_dual_axis(store, source, name, name)
history = Path("/source/.cache/dual-axis-prepared-001/prepared")
import_dual_axis(store, history, "history", "historical-index")
api = BasinAPI(store.root)
queries = [("basin_history", {}), ("basin_timeline", {"run_id": "deviation", "axis": "probe"}),
           ("basin_locate_deviation", {"run_id": "deviation"}),
           ("basin_branch_compare", {"left": "control", "right": "deviation"}),
           ("basin_verify", {"run_id": "deviation"})]
responses = [api.call(name, args) for name, args in queries]
alarm = responses[2]["data"]["deviations"]["items"][0]
assert alarm["possible_start_interval"] == [64, 128]
assert alarm["confirmation"]["coordinate"] == 129 and alarm["recovery"]["coordinate"] == 130
assert alarm["state"] == "recovered_after_persistent"
messages = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "dual-axis-qa", "version": "1"}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
messages += [{"jsonrpc": "2.0", "id": i + 3, "method": "tools/call", "params": {"name": name, "arguments": args}}
             for i, (name, args) in enumerate(queries)]
proc = subprocess.run([sys.executable, "-m", "basin", "--store", str(store.root), "mcp"],
                      input="".join(json.dumps(m) + "\n" for m in messages), text=True,
                      capture_output=True, timeout=30, check=True)
protocol = [json.loads(line) for line in proc.stdout.splitlines()]
assert len(protocol[1]["result"]["tools"]) == 12
assert all("error" not in r and not r.get("result", {}).get("isError") for r in protocol)
cli = subprocess.run([sys.executable, "-m", "basin", "--store", str(store.root), "call", "basin_locate_deviation"],
                     input=json.dumps({"run_id": "deviation"}), capture_output=True, text=True, timeout=30, check=True)
assert json.loads(cli.stdout) == responses[2]
for name, data in (("queries.json", responses), ("mcp-protocol.json", protocol),
                   ("verification.json", {"status": "passed", "kind": "synthetic", "tools": 12,
                                          "interval": [64, 128], "confirmed_at": 129, "recovered_at": 130,
                                          "model_execution": False, "real_model_overhead_measured": False})):
    with (output / name).open("x") as stream:
        json.dump(data, stream, indent=2)
print(json.dumps({"status": "passed", "protocol_responses": len(protocol), "tools": 12}))

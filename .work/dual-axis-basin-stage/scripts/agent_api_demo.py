"""A model-host-equivalent MCP client. No LLM/key required; no HTML involved."""

import argparse
import json
import queue
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basin.io import dumps, write_new


def run(store, output=None, command=None):
    command = command or [sys.executable, "-m", "basin", "--store", store, "mcp"]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, encoding="utf-8")
    messages = queue.Queue()
    errors = []
    def reader():
        for line in process.stdout:
            messages.put(line)
        messages.put(None)
    def stderr_reader():
        for line in process.stderr:
            if len(errors) < 20:
                errors.append(line)
    threading.Thread(target=reader, daemon=True).start()
    threading.Thread(target=stderr_reader, daemon=True).start()
    evidence, next_id = [], 0
    def request(method, params=None):
        nonlocal next_id
        next_id += 1
        msg = {"jsonrpc": "2.0", "id": next_id, "method": method, "params": params or {}}
        process.stdin.write(json.dumps(msg) + "\n")
        process.stdin.flush()
        line = messages.get(timeout=30)
        if line is None:
            raise RuntimeError("Server exited: " + "".join(errors))
        response = json.loads(line)
        if response.get("id") != next_id or "error" in response:
            raise RuntimeError(f"Protocol failure: {response}")
        result = response["result"]
        if result.get("isError"):
            raise RuntimeError(f"Tool failure: {result}")
        evidence.append({"request": msg, "response": response})
        return result
    def call(tool_name, **arguments):
        return request("tools/call", {"name": tool_name, "arguments": arguments})["structuredContent"]["data"]
    try:
        initialize = request("initialize", {"protocolVersion": "2025-11-25", "capabilities": {},
                                            "clientInfo": {"name": "basin-agent-api-demo", "version": "1"}})
        if initialize["protocolVersion"] != "2025-11-25":
            raise RuntimeError("Unsupported negotiated protocol")
        process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        process.stdin.flush()
        catalog = request("tools/list")
        history = call("basin_history", source_run="canonical-fullframes-posttrain-20260916-005", limit=20)
        parameters = call("basin_get_run", run_id="rosetta-gate4-1002", pointer="/parameters/options")
        event = call("basin_events", run_id="rosetta-gate4-1002", step=125, field="/action")
        row = event["items"][0]
        raw = call("basin_read_artifact", run_id="rosetta-gate4-1002", name=row["evidence"]["artifact"],
                   line=row["evidence"]["line"], pointer="/executed_action")
        if raw["value"] != row["value"]:
            raise AssertionError("Event and raw evidence disagree")
        analysis = call("basin_analyze", run_id="rosetta-gate4-1002", prefix="execution/action/", limit=20)
        diff = call("basin_compare", left="rosetta-gate4-1000", right="rosetta-gate4-1002", field="/action")
        verification = call("basin_verify", run_id="rosetta-gate4-1002")
        result = {"protocol": initialize["protocolVersion"], "tool_count": len(catalog["tools"]),
                  "history_units": history["total"], "queried_seed": parameters["seed"],
                  "queried_step": row["step"], "raw_action_matches_event": True,
                  "recorded_steps": analysis["facts"]["completed_execution_steps"],
                  "compared_events": diff["compared_events"], "integrity": verification["integrity"],
                  "llm_inference_used": False, "rosetta_modified": False, "transcript": evidence}
        if output:
            write_new(output, dumps(result))
        return {k: v for k, v in result.items() if k != "transcript"}
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="outputs/rosetta-demo-v2/history")
    parser.add_argument("--output", default="outputs/mcp-api-demo.json")
    parser.add_argument("--server-command", help="JSON argv array; use to test a host's exact launcher")
    parser.add_argument("--mcp-config", help="Read the basin entry of a trusted local mcpServers config")
    args = parser.parse_args()
    command = json.loads(args.server_command) if args.server_command else None
    if args.mcp_config:
        if command:
            parser.error("Choose either --server-command or --mcp-config")
        config = json.loads(Path(args.mcp_config).read_text(encoding="utf-8"))["mcpServers"]["basin"]
        command = [config["command"], *config.get("args", [])]
    print(dumps(run(args.store, args.output, command)))

"""Run from the Basin checkout: python -m basin --help."""

import argparse
import sys

from .adapters import import_gate, import_native, import_trace
from .analysis import analyze, compare, pointer
from .io import child, dumps, loads, read_bytes
from .report import render
from .store import Store
from .api import BasinAPI, ToolError, error_result, parse_sources
from .torchlens_data import analyze_torchlens, import_torchlens


def main(argv=None):
    parser = argparse.ArgumentParser(description="Basin: query evidence before changing a model")
    parser.add_argument("--store", default=".basin/history", help="Basin-owned history directory")
    parser.add_argument("--source", action="append", default=[], metavar="ALIAS=DIR",
                        help="Enable create-only API imports from this operator-configured root")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("tools", help="Discover model-callable tools and JSON input schemas")
    p.add_argument("--format", choices=("mcp", "functions"), default="mcp")
    p = sub.add_parser("call", help="Invoke a model tool with JSON arguments; stdout is JSON")
    p.add_argument("tool")
    p.add_argument("--args", default="-", help="JSON argument object; '-' reads UTF-8 JSON from stdin")
    sub.add_parser("mcp", help="Serve MCP 2025-11-25 JSON-RPC over stdio; no web server")
    sub.add_parser("history", help="List imports and integrity status")
    p = sub.add_parser("import-rosetta", help="Read-only import of existing Rosetta evidence")
    p.add_argument("repo")
    p.add_argument("path", help="Path relative to Rosetta root")
    p.add_argument("--id", required=True)
    p.add_argument("--kind", choices=("trace", "gate-report"), default="trace")
    p.add_argument("--gate", choices=("diagnostic", "gate3", "gate4"), default="diagnostic")
    p = sub.add_parser("import-native", help="Import a backend-neutral collector envelope")
    p.add_argument("path")
    p.add_argument("--id", required=True)
    p = sub.add_parser("import-torchlens", help="Import a Basin TorchLens JSON export")
    p.add_argument("path")
    p.add_argument("--id", required=True)
    p = sub.add_parser("analyze-torchlens", help="Analyze saved activations and observed graph references")
    p.add_argument("id")
    for command in ("show", "analyze", "verify", "events", "artifact"):
        p = sub.add_parser(command)
        p.add_argument("id")
        if command == "show":
            p.add_argument("--pointer", default="", help="JSON Pointer, e.g. /parameters/options/seed")
        if command == "events":
            p.add_argument("--step", type=int)
            p.add_argument("--stage")
            p.add_argument("--limit", type=int, default=20)
        if command == "artifact":
            p.add_argument("name")
            p.add_argument("--line", type=int)
            p.add_argument("--pointer", default="")
    p = sub.add_parser("compare")
    p.add_argument("left")
    p.add_argument("right")
    p.add_argument("--field", default="/action", help="JSON Pointer within event values")
    p.add_argument("--atol", type=float, default=0.0)
    p = sub.add_parser("report")
    p.add_argument("output", help="New HTML file (never overwrites)")
    p.add_argument("--compare", nargs=2, metavar=("LEFT", "RIGHT"))
    p.add_argument("--field", default="/action")
    args = parser.parse_args(argv)
    store = Store(args.store)
    try:
        if args.command in ("tools", "call", "mcp"):
            api = BasinAPI(args.store, parse_sources(args.source))
            if args.command == "mcp":
                from .mcp import serve
                return serve(api)
            if args.command == "tools":
                result = {"tools": api.tools()}
                if args.format == "functions":
                    result = {"tools": [{"type": "function", "function": {
                        "name": t["name"], "description": t["description"], "parameters": t["inputSchema"]}}
                        for t in api.tools()]}
            else:
                raw = sys.stdin.read(1024 * 1024 + 1) if args.args == "-" else args.args
                if len(raw.encode("utf-8")) > 1024 * 1024:
                    raise ToolError("invalid_arguments", "Arguments exceed 1 MiB")
                try:
                    arguments = loads(raw)
                except ValueError as exc:
                    raise ToolError("invalid_arguments", "Arguments must be valid JSON") from exc
                result = api.call(args.tool, arguments)
        elif args.command == "history":
            result = store.history()
        elif args.command == "import-rosetta":
            if args.kind == "trace":
                result = import_trace(store, args.repo, args.path, args.id, gate=args.gate)
            else:
                result = import_gate(store, args.repo, args.path, args.id)
        elif args.command == "import-native":
            result = import_native(store, args.path, args.id)
        elif args.command == "import-torchlens":
            result = import_torchlens(store, args.path, args.id)
        elif args.command == "analyze-torchlens":
            result = analyze_torchlens(store, args.id)
        elif args.command == "verify":
            result = store.verify(args.id)
        elif args.command == "show":
            result = pointer(store.get(args.id)[0], args.pointer)
        elif args.command == "events":
            if args.limit < 1 or args.limit > 1000:
                raise ValueError("Limit must be between 1 and 1000")
            events = store.get(args.id)[1]
            selected = [e for e in events if (args.step is None or e["step"] == args.step)
                        and (args.stage is None or e["stage"] == args.stage)]
            result = {"matched": len(selected), "events": selected[:args.limit]}
        elif args.command == "artifact":
            manifest = store.verify(args.id)
            name = "artifacts/" + args.name
            if name not in manifest["files"]:
                raise ValueError("Artifact not in manifest")
            data = read_bytes(child(store.path(args.id), name))
            if args.line is not None:
                if args.line < 1:
                    raise ValueError("Line numbers start at 1")
                data = data.splitlines()[args.line - 1]
            result = pointer(loads(data), args.pointer)
        elif args.command == "analyze":
            result = analyze(store, args.id)
        elif args.command == "compare":
            result = compare(store, args.left, args.right, args.field, args.atol)
        else:
            comparison = compare(store, *args.compare, field=args.field) if args.compare else None
            result = render(store, args.output, comparison)
        print(dumps(result))
        return 0
    except ToolError as exc:
        print(dumps(error_result(getattr(args, "tool", None), exc)))
        return 2
    except (ValueError, OSError, KeyError, TypeError, IndexError, OverflowError) as exc:
        print(dumps({"error": str(exc), "command": args.command}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

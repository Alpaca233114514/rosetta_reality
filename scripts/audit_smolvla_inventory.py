"""Create a local AST/import inventory; an inventory is not a semantic audit pass."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    paths = set((root / "src/rosetta_reality/vla").rglob("*.py"))
    for path in (root / "scripts").glob("*.py"):
        if any(word in path.name for word in ("smolvla", "iris", "hestia", "visual_", "train_m2")):
            paths.add(path)
    # Resolve repository imports recursively, including generic helpers actually
    # imported by the VLA entrypoints. No modules are executed here.
    pending, records = list(paths), {}
    while pending:
        path = pending.pop()
        name = path.relative_to(root).as_posix()
        if name in records:
            continue
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=name)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                imports.add(node.module)
                imports.update(f"{node.module}.{alias.name}" for alias in node.names)
        local = set()
        for module in imports:
            relative = Path(*module.split("."))
            for base in (root / "src", root, root / "scripts"):
                for candidate in (
                    base / relative.with_suffix(".py"),
                    base / relative / "__init__.py",
                ):
                    if candidate.is_file() and candidate.resolve().is_relative_to(root.resolve()):
                        local.add(candidate.relative_to(root).as_posix())
                        pending.append(candidate)
        records[name] = {
            "sha256": digest(path),
            "lines": len(source.splitlines()),
            "top_level_functions": [
                n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ],
            "local_imports": sorted(local),
            "status": "ast_and_import_inventory_only",
            "semantic_review": "see manually curated audit ledger; inventory alone is not a pass",
        }
    return dict(sorted(records.items()))


def inventory_v2(root):
    """Inventory all source roots plus declared file references, without imports.

    The legacy inventory remains reproducible. This version does not assume
    filename keywords or absolute Python imports enumerate executable roots.
    Unresolved computed imports remain explicit review obligations.
    """
    root = Path(root).resolve()
    suffixes = {".py", ".sh", ".yaml", ".json", ".jsonl", ".md", ".txt", ".toml", ".cpp", ".xml"}
    paths = {root / name for name in ("AGENTS.md", "README.md", "pyproject.toml")}
    for folder in (
        "src",
        "scripts",
        "configs",
        "docs",
        "tests",
        "integration",
        "docker",
        "experiments",
    ):
        paths.update(p for p in (root / folder).rglob("*") if p.is_file())
    paths.update(p for p in (root / "reports/training").rglob("*") if p.is_file())
    records = {}
    reference = re.compile(
        r"(?:scripts|src|configs|docs|reports/training)/[A-Za-z0-9_./-]+\.(?:py|sh|yaml|json|md)"
    )
    for path in sorted(paths):
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            continue
        if path.suffix not in suffixes and path.parent.name != "docker":
            continue
        if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        name = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8-sig")
        imports, dynamic = set(), []
        if path.suffix == ".py":
            tree = ast.parse(source, filename=name)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.add("." * node.level + (node.module or ""))
                elif isinstance(node, ast.Call):
                    called = ast.unparse(node.func)
                    if any(
                        x in called
                        for x in (
                            "import_module",
                            "__import__",
                            "spec_from_file",
                            "run_path",
                            "Popen",
                            "subprocess.run",
                        )
                    ):
                        dynamic.append({"line": node.lineno, "expression": ast.unparse(node)[:400]})
        refs = sorted(set(reference.findall(source)))
        records[name] = {
            "sha256": digest(path),
            "bytes": path.stat().st_size,
            "lines": len(source.splitlines()),
            "imports": sorted(imports),
            "declared_file_references": refs,
            "dynamic_calls": dynamic,
            "missing_references": [p for p in refs if not (root / p).is_file()],
            "role": name.split("/")[0],
            "status": "indexed_not_semantically_verified",
            "semantic_review": "pending; enumeration is not a correctness proof",
        }
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    records = inventory(root)
    package = importlib.util.find_spec("lerobot")
    package_root = Path(next(iter(package.submodule_search_locations)))
    upstream = {}
    for name in (
        "scripts/lerobot_train.py",
        "datasets/sampler.py",
        "optim/schedulers.py",
        "optim/optimizers.py",
        "policies/smolvla/modeling_smolvla.py",
        "policies/smolvla/smolvlm_with_expert.py",
        "utils/train_utils.py",
    ):
        path = package_root / name
        if path.is_file():
            upstream[name] = {"sha256": digest(path), "source": path.read_text()}
        else:
            upstream[name] = {"status": "missing"}
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "inventory.json").open("x") as stream:
        json.dump(
            {
                "status": "inventory_complete_not_semantic_acceptance",
                "files": records,
                "file_count": len(records),
                "upstream": upstream,
                "model_loaded": False,
                "optimizer_steps": 0,
            },
            stream,
            indent=2,
        )
    print(json.dumps({"files": len(records), "upstream_files": len(upstream)}))


if __name__ == "__main__":
    main()

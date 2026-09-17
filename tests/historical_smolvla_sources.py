"""Build byte-verified historical protocol surfaces; never fake a digest result."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures/smolvla_historical_sources"


def bind_historical_sources(protocol, plan, destination, monkeypatch):
    inventory = dict(plan["implementation_files"])
    inventory.update(
        {
            protocol.PARENT_CONFIG: protocol.PARENT_SHA256,
            "configs/runtime/autodl_rtx4090.yaml": protocol.RUNTIME_PROFILE_SHA256,
            protocol.NORMALIZATION_REPORT_RELATIVE: protocol.NORMALIZATION_REPORT_SHA256,
            protocol.VIEW_MANIFEST_RELATIVE: protocol.VIEW_MANIFEST_SHA256,
        }
    )
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    for name, expected in inventory.items():
        original = ROOT / name
        content = original.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            entry = manifest[expected]
            assert entry["source_path"] == name
            content = (FIXTURES / f"{expected}.txt").read_bytes()
            assert len(content) == entry["bytes"]
        assert hashlib.sha256(content).hexdigest() == expected, name
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    monkeypatch.setattr(
        protocol, "__file__", str(destination / "scripts" / Path(protocol.__file__).name)
    )
    return destination

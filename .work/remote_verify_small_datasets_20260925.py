import hashlib
import json
from pathlib import Path
import urllib.request


root = Path("/root/autodl-tmp/rosetta/datasets_external/targeted-20260925-001")
sources = (
    ("lerobot/aloha_sim_insertion_scripted", "8ab660912970111cbb26738b11458e6fc4a4aed1", 10, 70994713),
    ("robomimic/robomimic_datasets", "74fa018461f479cd9fd15b924a16103012096203", 2, 100568305),
)
for repo, revision, expected_count, expected_bytes in sources:
    target = root / repo / revision
    manifest = json.loads((target / "rosetta_download_manifest.json").read_text())
    assert manifest["repo"] == repo and manifest["revision"] == revision
    assert len(manifest["files"]) == expected_count
    api = "https://hf-mirror.com/api/datasets/" + repo + "/tree/" + revision + "?recursive=true&expand=true"
    with urllib.request.urlopen(urllib.request.Request(api, headers={"User-Agent": "rosetta-dataset-verifier/1"}), timeout=45) as response:
        official = {item["path"]: item for item in json.load(response) if item.get("type") == "file"}
    total = 0
    for item in manifest["files"]:
        path = target / item["path"]
        assert path.is_file() and not path.is_symlink()
        assert item["path"] in official
        assert path.stat().st_size == item["bytes"] == official[item["path"]]["size"]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
        actual = digest.hexdigest()
        assert actual == item["sha256"]
        lfs = official[item["path"]].get("lfs") or {}
        if lfs.get("oid"):
            assert actual == lfs["oid"]
        total += item["bytes"]
    assert total == expected_bytes
    print(json.dumps({"repo": repo, "revision": revision, "files": expected_count, "bytes": total, "verified": True}))

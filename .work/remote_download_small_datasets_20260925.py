import hashlib
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request
import uuid


ROOT = Path("/root/autodl-tmp/rosetta/datasets_external/targeted-20260925-001")
BASE = "https://hf-mirror.com"
SOURCES = (
    (
        "lerobot/aloha_sim_insertion_scripted",
        "8ab660912970111cbb26738b11458e6fc4a4aed1",
        "all",
    ),
    (
        "robomimic/robomimic_datasets",
        "74fa018461f479cd9fd15b924a16103012096203",
        {"v1.5/can/paired/demo_v15.hdf5", "v1.5/can/paired/low_dim_v15.hdf5"},
    ),
)


def request(url):
    return urllib.request.Request(url, headers={"User-Agent": "rosetta-dataset-direct-download/1"})


def get_json(url):
    with urllib.request.urlopen(request(url), timeout=45) as response:
        return json.load(response)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download(repo, sha, item, destination):
    relative = item["path"]
    expected_size = item["size"]
    expected_sha = (item.get("lfs") or {}).get("oid")
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or target.stat().st_size != expected_size:
            raise RuntimeError("existing target has wrong type/size: " + str(target))
        actual_sha = sha256_file(target)
        if expected_sha and actual_sha != expected_sha:
            raise RuntimeError("existing target SHA mismatch: " + str(target))
        return {"path": relative, "bytes": expected_size, "sha256": actual_sha, "status": "verified_existing"}
    encoded_path = urllib.parse.quote(relative, safe="/")
    url = BASE + "/datasets/" + repo + "/resolve/" + sha + "/" + encoded_path
    for attempt in range(1, 4):
        partial = target.with_name(target.name + ".partial-" + uuid.uuid4().hex)
        digest = hashlib.sha256()
        count = 0
        try:
            with urllib.request.urlopen(request(url), timeout=60) as response, partial.open("xb") as output:
                while block := response.read(8 * 1024 * 1024):
                    output.write(block)
                    digest.update(block)
                    count += len(block)
            actual_sha = digest.hexdigest()
            if count != expected_size:
                raise RuntimeError(f"size mismatch {relative}: {count} != {expected_size}")
            if expected_sha and actual_sha != expected_sha:
                raise RuntimeError(f"SHA mismatch {relative}: {actual_sha} != {expected_sha}")
            os.rename(partial, target)
            return {"path": relative, "bytes": count, "sha256": actual_sha, "status": "downloaded"}
        except Exception as exc:
            print(json.dumps({"file": relative, "attempt": attempt, "error": str(exc), "partial_preserved": str(partial)}), flush=True)
            if attempt == 3:
                raise
            time.sleep(attempt * 2)


ROOT.mkdir(parents=True, exist_ok=True)
for repo, sha, selection in SOURCES:
    data = get_json(BASE + "/api/datasets/" + repo)
    if data.get("sha") != sha:
        raise RuntimeError("repository revision drift: " + repo)
    entries = get_json(BASE + "/api/datasets/" + repo + "/tree/" + sha + "?recursive=true&expand=true")
    files = [item for item in entries if item.get("type") == "file"]
    if selection == "all":
        if len(files) != 10:
            raise RuntimeError("unexpected ALOHA file count")
        chosen = files
    else:
        chosen = [item for item in files if item["path"] in selection]
        if {item["path"] for item in chosen} != selection:
            raise RuntimeError("Can Paired selection incomplete")
    destination = ROOT / repo / sha
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for item in chosen:
        result = download(repo, sha, item, destination)
        records.append(result)
        print(json.dumps({"repo": repo, **result}), flush=True)
    manifest = destination / "rosetta_download_manifest.json"
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump({"repo": repo, "revision": sha, "mirror": BASE, "files": records}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"complete": repo, "file_count": len(records), "bytes": sum(record["bytes"] for record in records), "manifest": str(manifest)}), flush=True)

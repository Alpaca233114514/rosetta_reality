"""Independent remote rehash for the direct-download dataset receipts."""

import base64
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request


ROOT = Path("/root/autodl-tmp/rosetta/datasets_external/targeted-20260925-001")
MIRROR = "https://hf-mirror.com"
SOURCES = (
    ("REBOOT26/USB-A_recovery_install", "cfa3498a3982eb24554e88e77140247871eee3eb", 82, 37283634402),
    ("REBOOT26/rj45_recovery_install", "f6acd5b69394ffd4d2d8bd30079306c9e2061bbe", 83, 37829542278),
    ("amandlek/mimicgen_datasets", "33016f8a62c02334f929f2913af8fdd2a8a129e1", 3, 4851394230),
)


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "rosetta-dataset-independent-verifier/1"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response), response.headers.get("Link", "")


def official_tree(repo, revision):
    url = f"{MIRROR}/api/datasets/{repo}/tree/{revision}?recursive=true&expand=true"
    output = {}
    while url:
        entries, link = get_json(url)
        for entry in entries:
            if entry.get("type") == "file":
                if entry["path"] in output:
                    raise RuntimeError("duplicate source path")
                output[entry["path"]] = entry
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part.split("<", 1)[1].split(">", 1)[0].replace("https://huggingface.co/", MIRROR + "/")
    return output


def digests(path):
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), base64.b64encode(md5.digest()).decode("ascii")


for repo, revision, expected_count, expected_bytes in SOURCES:
    target = ROOT / repo / revision
    receipt = json.loads((target / "rosetta_download_manifest.json").read_text(encoding="utf-8"))
    assert receipt["repo"] == repo and receipt["revision"] == revision
    assert len(receipt["files"]) == expected_count
    upstream = official_tree(repo, revision)
    total = 0
    for item in receipt["files"]:
        relative = item["path"]
        assert relative in upstream
        source = upstream[relative]
        path = target / relative
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == item["bytes"] == source["size"]
        sha, md5 = digests(path)
        assert sha == item["sha256"]
        assert md5 == item["md5_base64"]
        source_lfs = source.get("lfs") or {}
        if source_lfs.get("oid"):
            assert sha == source_lfs["oid"]
        total += path.stat().st_size
    assert total == expected_bytes
    print(json.dumps({"repo": repo, "revision": revision, "files": expected_count, "bytes": total, "verified": True}), flush=True)


prefix = "austin_sirius_dataset_converted_externally_to_rlds/"
target = ROOT / "open_x_embodiment" / "austin_sirius_dataset_converted_externally_to_rlds"
receipt = json.loads((target / "rosetta_download_manifest.json").read_text(encoding="utf-8"))
assert receipt["prefix"] == prefix and len(receipt["files"]) == 67
gcs_api = "https://storage.googleapis.com/storage/v1/b/gdm-robotics-open-x-embodiment/o?" + urllib.parse.urlencode({"prefix": prefix, "maxResults": 1000})
official_objects, _ = get_json(gcs_api)
assert not official_objects.get("nextPageToken")
official_by_path = {item["name"][len(prefix):]: item for item in official_objects["items"]}
assert len(official_by_path) == 67
total = 0
for item in receipt["files"]:
    official = official_by_path[item["path"]]
    assert official["generation"] == item["generation"]
    assert int(official["size"]) == item["source_size"]
    assert official["md5Hash"] == item["source_md5_base64"]
    path = target / item["path"]
    assert path.is_file() and not path.is_symlink()
    assert path.stat().st_size == item["bytes"] == item["source_size"]
    sha, md5 = digests(path)
    assert sha == item["sha256"]
    assert md5 == item["md5_base64"] == item["source_md5_base64"]
    total += item["bytes"]
assert total == 7031694115
print(json.dumps({"repo": "sirius", "files": 67, "bytes": total, "verified": True}), flush=True)

"""Direct, revision-pinned downloads on the authorized AutoDL SSH host.

Stage this script on the remote host only after the checkpoint handoff and
the planned disk expansion have been verified. It never transfers dataset
bytes through the local machine.
"""

import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.parse
import urllib.request


ROOT = Path("/root/autodl-tmp/rosetta/datasets_external/targeted-20260925-001")
MIRROR = "https://hf-mirror.com"
GCS_BUCKET = "gdm-robotics-open-x-embodiment"
MIN_FREE_BYTES = 10_000_000_000
CHUNK_BYTES = 8 * 1024 * 1024
USER_AGENT = "rosetta-revision-pinned-dataset-download/1"
HF_SOURCES = (
    ("REBOOT26/USB-A_recovery_install", "cfa3498a3982eb24554e88e77140247871eee3eb", None),
    ("REBOOT26/rj45_recovery_install", "f6acd5b69394ffd4d2d8bd30079306c9e2061bbe", None),
    (
        "amandlek/mimicgen_datasets",
        "33016f8a62c02334f929f2913af8fdd2a8a129e1",
        {"core/square_d0.hdf5", "core/square_d1.hdf5", "core/square_d2.hdf5"},
    ),
)
HF_EXPECTED = {
    "REBOOT26/USB-A_recovery_install": (82, 37283634402),
    "REBOOT26/rj45_recovery_install": (83, 37829542278),
    "amandlek/mimicgen_datasets": (3, 4851394230),
}


def emit(**event):
    print(json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)


def urlopen(url, timeout=90):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=timeout)


def get_json(url):
    for attempt in range(1, 5):
        try:
            with urlopen(url, timeout=45) as response:
                return json.load(response)
        except Exception as exc:
            emit(status="source_retry", attempt=attempt, error=str(exc))
            if attempt == 4:
                raise
            time.sleep(attempt * 2)


def file_digests(path):
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as stream:
        while block := stream.read(CHUNK_BYTES):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), base64.b64encode(md5.digest()).decode("ascii")


def free_bytes():
    return shutil.disk_usage(ROOT).free


def tree_pages(repo, revision):
    encoded = urllib.parse.quote(repo, safe="/")
    url = f"{MIRROR}/api/datasets/{encoded}/tree/{revision}?recursive=true&expand=true"
    entries = []
    while url:
        for attempt in range(1, 5):
            try:
                with urlopen(url, timeout=45) as response:
                    page = json.load(response)
                    link = response.headers.get("Link", "")
                break
            except Exception as exc:
                emit(status="source_retry", attempt=attempt, error=str(exc))
                if attempt == 4:
                    raise
                time.sleep(attempt * 2)
        entries.extend(page)
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part.split("<", 1)[1].split(">", 1)[0].replace("https://huggingface.co/", MIRROR + "/")
    return [entry for entry in entries if entry.get("type") == "file"]


def download(url, target, expected_size, expected_sha=None, expected_md5=None):
    if target.is_symlink():
        raise RuntimeError("target is symlink: " + str(target))
    if target.exists():
        if not target.is_file() or target.stat().st_size != expected_size:
            raise RuntimeError("existing target size/type mismatch: " + str(target))
        sha, md5 = file_digests(target)
        if expected_sha and sha != expected_sha:
            raise RuntimeError("existing target SHA mismatch: " + str(target))
        if expected_md5 and md5 != expected_md5:
            raise RuntimeError("existing target MD5 mismatch: " + str(target))
        return {"bytes": expected_size, "sha256": sha, "md5_base64": md5, "status": "verified_existing"}
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".inprogress-20260925-001")
    if partial.is_symlink():
        raise RuntimeError("partial is symlink: " + str(partial))
    if partial.exists() and partial.stat().st_size > expected_size:
        raise RuntimeError("partial larger than source: " + str(partial))
    for attempt in range(1, 7):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected_size:
            break
        if free_bytes() < MIN_FREE_BYTES:
            raise RuntimeError("free disk below safety reserve before " + str(target))
        headers = {"User-Agent": USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                if offset and response.status != 206:
                    raise RuntimeError(f"range resume refused ({response.status}); partial preserved")
                if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("range offset mismatch; partial preserved")
                with partial.open("ab" if offset else "xb") as output:
                    while block := response.read(CHUNK_BYTES):
                        output.write(block)
            if partial.stat().st_size != expected_size:
                raise RuntimeError("incomplete response")
        except Exception as exc:
            emit(status="retry", target=str(target), attempt=attempt, partial_bytes=partial.stat().st_size if partial.exists() else 0, error=str(exc))
            if attempt == 6:
                raise
            time.sleep(min(30, attempt * 3))
    if not partial.exists() or partial.stat().st_size != expected_size:
        raise RuntimeError("incomplete download: " + str(partial))
    sha, md5 = file_digests(partial)
    if expected_sha and sha != expected_sha:
        raise RuntimeError("downloaded SHA mismatch; partial preserved: " + str(partial))
    if expected_md5 and md5 != expected_md5:
        raise RuntimeError("downloaded MD5 mismatch; partial preserved: " + str(partial))
    os.link(partial, target)
    partial.unlink()
    return {"bytes": expected_size, "sha256": sha, "md5_base64": md5, "status": "downloaded"}


def save_manifest(destination, payload):
    path = destination / "rosetta_download_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        def without_run_status(document):
            result = dict(document)
            result["files"] = [{key: value for key, value in item.items() if key != "status"} for item in document["files"]]
            return result
        if without_run_status(existing) != without_run_status(payload):
            raise RuntimeError("existing manifest differs: " + str(path))
        return
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def download_hf(repo, revision, selection):
    info = get_json(f"{MIRROR}/api/datasets/{repo}")
    all_files = tree_pages(repo, revision)
    files = [item for item in all_files if selection is None or item["path"] in selection]
    if selection is not None and {item["path"] for item in files} != selection:
        raise RuntimeError("HF file selection incomplete: " + repo)
    expected_bytes = sum(int(item["size"]) for item in files)
    if (len(files), expected_bytes) != HF_EXPECTED[repo]:
        raise RuntimeError("pinned HF file inventory drift: " + repo)
    destination = ROOT / repo / revision
    destination.mkdir(parents=True, exist_ok=True)
    emit(status="source", repo=repo, revision=revision, current_head=info.get("sha"), files=len(files), bytes=expected_bytes)
    records = []
    for item in files:
        relative = item["path"]
        if not relative or relative.startswith("/") or ".." in Path(relative).parts:
            raise RuntimeError("unsafe HF source path: " + relative)
        lfs = item.get("lfs") or {}
        url = f"{MIRROR}/datasets/{repo}/resolve/{revision}/" + urllib.parse.quote(relative, safe="/")
        result = download(url, destination / relative, int(item["size"]), expected_sha=lfs.get("oid"))
        records.append({"path": relative, "source_size": item["size"], "source_lfs_sha256": lfs.get("oid"), **result})
        emit(status="file_complete", repo=repo, path=relative, bytes=result["bytes"], sha256=result["sha256"])
    payload = {"source_type": "huggingface_dataset", "repo": repo, "revision": revision, "source_endpoint": MIRROR, "files": records}
    save_manifest(destination, payload)
    emit(status="repo_complete", repo=repo, files=len(records), bytes=expected_bytes)


def gcs_objects(prefix):
    url = "https://storage.googleapis.com/storage/v1/b/" + GCS_BUCKET + "/o?" + urllib.parse.urlencode({"prefix": prefix, "maxResults": 1000})
    while url:
        data = get_json(url)
        yield from data.get("items", [])
        token = data.get("nextPageToken")
        url = "https://storage.googleapis.com/storage/v1/b/" + GCS_BUCKET + "/o?" + urllib.parse.urlencode({"prefix": prefix, "maxResults": 1000, "pageToken": token}) if token else None


def download_sirius():
    prefix = "austin_sirius_dataset_converted_externally_to_rlds/"
    objects = list(gcs_objects(prefix))
    if len(objects) != 67 or sum(int(item["size"]) for item in objects) != 7031694115:
        raise RuntimeError("Sirius GCS object inventory drift")
    destination = ROOT / "open_x_embodiment" / "austin_sirius_dataset_converted_externally_to_rlds"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for item in objects:
        relative = item["name"][len(prefix):]
        if not relative or relative.startswith("/") or ".." in Path(relative).parts:
            raise RuntimeError("unsafe GCS object path")
        object_url = "https://storage.googleapis.com/" + GCS_BUCKET + "/" + urllib.parse.quote(item["name"], safe="/") + "?generation=" + item["generation"]
        result = download(object_url, destination / relative, int(item["size"]), expected_md5=item.get("md5Hash"))
        records.append({"path": relative, "generation": item["generation"], "source_size": int(item["size"]), "source_md5_base64": item.get("md5Hash"), **result})
        emit(status="file_complete", repo="sirius", path=relative, bytes=result["bytes"], sha256=result["sha256"])
    payload = {"source_type": "gcs_open_x_embodiment", "bucket": GCS_BUCKET, "prefix": prefix, "files": records}
    save_manifest(destination, payload)
    emit(status="repo_complete", repo="sirius", files=len(records), bytes=sum(item["bytes"] for item in records))


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if free_bytes() < 95_000_000_000:
        raise RuntimeError("planned complete payload requires expanded data disk; free space below 95 GB")
    emit(status="start", destination=str(ROOT), free_bytes=free_bytes())
    for repo, revision, selection in HF_SOURCES:
        download_hf(repo, revision, selection)
    download_sirius()
    emit(status="all_complete", free_bytes=free_bytes())


if __name__ == "__main__":
    main()

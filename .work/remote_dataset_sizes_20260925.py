import json
import urllib.parse
import urllib.request


BASE = "https://hf-mirror.com/api/datasets/"


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "rosetta-dataset-preflight/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


for repo_id in (
    "REBOOT26/USB-A_recovery_install",
    "REBOOT26/rj45_recovery_install",
    "lerobot/aloha_sim_insertion_scripted",
    "amandlek/mimicgen_datasets",
    "robomimic/robomimic_datasets",
):
    info = get_json(BASE + urllib.parse.quote(repo_id, safe="/"))
    sha = info["sha"]
    tree = get_json(BASE + urllib.parse.quote(repo_id, safe="/") + "/tree/" + sha + "?recursive=true&expand=true")
    files = [entry for entry in tree if entry.get("type") == "file"]
    total = sum(entry.get("size") or 0 for entry in files)
    selected = [
        {"path": entry["path"], "size": entry.get("size"), "oid": entry.get("lfs", {}).get("oid")}
        for entry in files
        if (repo_id not in ("amandlek/mimicgen_datasets", "robomimic/robomimic_datasets"))
        or entry["path"] in ("core/square_d0.hdf5", "core/square_d1.hdf5", "core/square_d2.hdf5", "v1.5/can/paired/low_dim_v15.hdf5", "v1.5/can/paired/demo_v15.hdf5")
    ]
    print(json.dumps({"id": repo_id, "sha": sha, "file_count": len(files), "total_bytes": total, "selected_bytes": sum(item["size"] or 0 for item in selected), "selected": selected if len(selected) <= 10 else selected[:3] + selected[-3:]}, ensure_ascii=False))

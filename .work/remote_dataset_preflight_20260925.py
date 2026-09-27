import json
import urllib.error
import urllib.parse
import urllib.request


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "rosetta-dataset-preflight/1"})
    with urllib.request.urlopen(request, timeout=25) as response:
        return json.load(response)


def inspect(repo_id, prefixes=()):
    url = "https://hf-mirror.com/api/datasets/" + urllib.parse.quote(repo_id, safe="/")
    try:
        data = get_json(url)
    except Exception as exc:
        print(json.dumps({"id": repo_id, "error": str(exc)}))
        return
    files = data.get("siblings", [])
    chosen = []
    for item in files:
        name = item.get("rfilename", "")
        if not prefixes or any(name.startswith(prefix) for prefix in prefixes):
            chosen.append({"path": name, "size": item.get("size"), "lfs": item.get("lfs", {}).get("size")})
    print(json.dumps({"id": repo_id, "sha": data.get("sha"), "private": data.get("private"), "gated": data.get("gated"), "total_siblings": len(files), "files": chosen}, ensure_ascii=False))


authors = get_json("https://hf-mirror.com/api/datasets?author=REBOOT26&limit=100")
print(json.dumps({"author": "REBOOT26", "repos": [item.get("id") for item in authors]}, ensure_ascii=False))
inspect("REBOOT26/USB-A_recovery_install")
inspect("REBOOT26/rj45_recovery_install")
inspect("lerobot/aloha_sim_insertion_scripted")
inspect("amandlek/mimicgen_datasets", ("core/square_d0", "core/square_d1", "core/square_d2", "README", ".gitattributes"))

import json
import urllib.parse
import urllib.request


for repo_id in ("REBOOT26/USB-A_recovery_install", "REBOOT26/rj45_recovery_install"):
    encoded = urllib.parse.quote(repo_id, safe="/")
    with urllib.request.urlopen(urllib.request.Request("https://hf-mirror.com/api/datasets/" + encoded, headers={"User-Agent": "rosetta-dataset-preflight/1"}), timeout=30) as response:
        info = json.load(response)
    url = "https://hf-mirror.com/api/datasets/" + encoded + "/tree/" + info["sha"] + "?recursive=true&expand=true"
    files = []
    pages = 0
    while url:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "rosetta-dataset-preflight/1"}), timeout=30) as response:
            entries = json.load(response)
            link = response.headers.get("Link", "")
        pages += 1
        files.extend(item for item in entries if item.get("type") == "file")
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part.split("<", 1)[1].split(">", 1)[0].replace("https://huggingface.co/", "https://hf-mirror.com/")
    groups = {}
    for item in files:
        group = item["path"].split("/", 1)[0]
        groups[group] = groups.get(group, 0) + (item.get("size") or 0)
    print(json.dumps({"repo": repo_id, "sha": info["sha"], "pages": pages, "files": len(files), "bytes": sum(item.get("size") or 0 for item in files), "groups": groups}))

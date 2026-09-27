import json
import urllib.request


for repo, sha in (
    ("REBOOT26/USB-A_recovery_install", "cfa3498a3982eb24554e88e77140247871eee3eb"),
    ("REBOOT26/rj45_recovery_install", "f6acd5b69394ffd4d2d8bd30079306c9e2061bbe"),
):
    base = "https://hf-mirror.com/datasets/" + repo + "/resolve/" + sha + "/"
    with urllib.request.urlopen(urllib.request.Request(base + "meta/info.json", headers={"User-Agent": "rosetta-reboot-preflight/1"}), timeout=30) as response:
        info = json.load(response)
    with urllib.request.urlopen(urllib.request.Request(base + "README.md", headers={"User-Agent": "rosetta-reboot-preflight/1"}), timeout=30) as response:
        readme = response.read().decode("utf-8", "replace")
    print(json.dumps({"repo": repo, "sha": sha, "episodes": info.get("total_episodes"), "frames": info.get("total_frames"), "fps": info.get("fps"), "video_path": info.get("video_path"), "data_path": info.get("data_path"), "readme_excerpt": readme[:2300]}, ensure_ascii=False))

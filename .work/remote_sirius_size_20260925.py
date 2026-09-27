import json
import urllib.parse
import urllib.request


bucket = "gdm-robotics-open-x-embodiment"
prefix = "austin_sirius_dataset_converted_externally_to_rlds/"
url = "https://storage.googleapis.com/storage/v1/b/" + bucket + "/o?" + urllib.parse.urlencode({"prefix": prefix, "maxResults": 1000})
count = 0
total = 0
missing_md5 = 0
examples = []
while url:
    request = urllib.request.Request(url, headers={"User-Agent": "rosetta-dataset-preflight/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)
    for item in data.get("items", []):
        count += 1
        total += int(item["size"])
        missing_md5 += not bool(item.get("md5Hash"))
        if len(examples) < 3:
            examples.append({"name": item["name"], "size": item["size"], "md5Hash": item.get("md5Hash")})
    token = data.get("nextPageToken")
    url = "https://storage.googleapis.com/storage/v1/b/" + bucket + "/o?" + urllib.parse.urlencode({"prefix": prefix, "maxResults": 1000, "pageToken": token}) if token else None
print(json.dumps({"prefix": prefix, "file_count": count, "total_bytes": total, "missing_md5": missing_md5, "examples": examples}))

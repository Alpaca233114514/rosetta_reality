import json
import urllib.parse
import urllib.request


bucket = "gdm-robotics-open-x-embodiment"
prefix = "austin_sirius_dataset_converted_externally_to_rlds/"
api = "https://storage.googleapis.com/storage/v1/b/" + bucket + "/o?" + urllib.parse.urlencode({"prefix": prefix, "maxResults": 1})
with urllib.request.urlopen(api, timeout=30) as response:
    item = json.load(response)["items"][0]
url = "https://storage.googleapis.com/" + bucket + "/" + urllib.parse.quote(item["name"], safe="/") + "?generation=" + item["generation"]
request = urllib.request.Request(url, method="HEAD")
with urllib.request.urlopen(request, timeout=30) as response:
    print(json.dumps({"name": item["name"], "generation": item["generation"], "status": response.status, "content_length": int(response.headers["Content-Length"]), "expected_size": int(item["size"])}))

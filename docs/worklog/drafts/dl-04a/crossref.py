import json, subprocess, sys

for doi in sys.argv[1:]:
    out = subprocess.run(["curl", "-s", "--max-time", "25", f"https://api.crossref.org/works/{doi}"],
                         capture_output=True, text=True).stdout
    m = json.loads(out)["message"]
    print(m["title"][0], "|", m.get("container-title"), "|", m["issued"]["date-parts"][0][0])
code = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                       "https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/PixelShuffle.cpp"],
                      capture_output=True, text=True).stdout
print("PixelShuffle.cpp raw:", code)

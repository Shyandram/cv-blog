import sys, time, urllib.request, xml.etree.ElementTree as ET
ns = {"a": "http://www.w3.org/2005/Atom"}
ids = sys.argv[1:]
url = "https://export.arxiv.org/api/query?max_results=50&id_list=" + ",".join(ids)
for attempt in range(5):
    try:
        data = urllib.request.urlopen(url, timeout=60).read()
        root = ET.fromstring(data)
        break
    except Exception as e:
        print("retry", e, file=sys.stderr); time.sleep(6)
for e in root.findall("a:entry", ns):
    print(e.find("a:id", ns).text, "|", " ".join(e.find("a:title", ns).text.split()))
    print("   ", " ".join(e.find("a:summary", ns).text.split()), "\n")

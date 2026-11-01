import re, sys
sys.path.insert(0, "/tmp/claude-0/-home-user-cv-blog/46b65cc4-04fc-53dc-9c17-ea7809aadfc9/scratchpad/sr04a_code")

src = open(sys.argv[1], encoding="utf-8").read()

def clean(t):
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"\$\$.*?\$\$", " ", t, flags=re.S)
    t = re.sub(r"<Figure[^>]*/>|<ImageCompare[^>]*/>", " ", t, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\$[^$\n]+\$", " ", t)
    t = re.sub(r"\]\([^)]*\)", "]", t)
    t = re.sub(r"\[\d+\]", " ", t)
    t = re.sub(r"`[^`]*`", " x ", t)
    return len(re.findall(r"[A-Za-z0-9][\w'’\-.]*", t))

parts = re.split(r"\n(?=## )", src)
for p in parts:
    print(f"{clean(p):5d}  {p.splitlines()[0][:70]}")

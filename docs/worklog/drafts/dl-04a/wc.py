import re, sys

src = open(sys.argv[1], encoding="utf-8").read()
src = src.split("\n---\n", 2)[-1] if src.startswith("---") else src
src = re.sub(r"^import .*$", "", src, flags=re.M)

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

main, rest = src.split("\n## Supplementary notes\n", 1)
notes, refs = rest.split("\n## References\n", 1)
ex_start = main.index("\n## Exercises\n")
print("main (incl. exercises):", clean(main))
print("main without exercises:", clean(main[:ex_start]))
print("exercises:", clean(main[ex_start:]))
print("notes:", clean(notes))

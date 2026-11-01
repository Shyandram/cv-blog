"""Turn [@key] citations and '@key: entry' reference lines into numbered citations (by first use)."""
import re, sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
body, refs = src.split("\n## References\n", 1)
entries = {}
for line in refs.strip().splitlines():
    m = re.match(r"@(\w+): (.*)$", line)
    if not m:
        raise SystemExit(f"bad reference line: {line!r}")
    entries[m.group(1)] = m.group(2)

order = []
for key in re.findall(r"\[@(\w+)\]", body):
    if key not in entries:
        raise SystemExit(f"citation without entry: {key}")
    if key not in order:
        order.append(key)
unused = [k for k in entries if k not in order]
if unused:
    raise SystemExit(f"entries never cited: {unused}")
num = {k: i + 1 for i, k in enumerate(order)}
body = re.sub(r"\[@(\w+)\]", lambda m: f"[{num[m.group(1)]}]", body)
ref_lines = [f"{num[k]}. {entries[k]}" for k in order]
open(path, "w", encoding="utf-8").write(body + "\n## References\n\n" + "\n".join(ref_lines) + "\n")
for k in order:
    print(num[k], k)

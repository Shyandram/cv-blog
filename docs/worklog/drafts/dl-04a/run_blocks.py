"""Run every ```python block of an MDX file and compare stdout with the ```text block that follows it."""
import re, subprocess, sys

src = open(sys.argv[1], encoding="utf-8").read()
blocks = re.findall(r"```(python|text)\n(.*?)```", src, flags=re.S)
ok = True
for i, (lang, code) in enumerate(blocks):
    if lang != "python":
        continue
    expected = blocks[i + 1][1] if i + 1 < len(blocks) and blocks[i + 1][0] == "text" else None
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    got = out.stdout
    match = expected is not None and got.strip() == expected.strip()
    ok &= match
    print(f"--- block {i}: {'MATCH' if match else 'DIFF'}")
    if not match:
        print("stderr:", out.stderr[-2000:])
        print("got:\n" + got)
        print("expected:\n" + str(expected))
print("ALL MATCH" if ok else "SOME DIFFER")

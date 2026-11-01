import sys
p = sys.argv[1]
s = open(p).read()
add = open(sys.argv[2]).read()
anchor = "\n/* Shiki dual themes"
assert s.count(anchor) == 1
s = s.replace(anchor, add + anchor)
old = "  figure, pre, table, .katex-display, details, blockquote, .callout { break-inside: avoid; }"
new = old + """
  .dis-act, .expand-all, .sm-tools { display: none !important; }
  :is(.disclosure, .prose details) > summary { background: none; min-height: 0; padding: 4px 0; }
  :is(.disclosure, .prose details) > summary::before { display: none; }
  .disclosure, .prose details { border: 0; }
  .dis-body { padding: 0; }"""
assert s.count(old) == 1
s = s.replace(old, new)
open(p, 'w').write(s)

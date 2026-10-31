// Turns numeric citations such as [3] or [1, 4] into links to the matching entry of the page's reference list, and gives the
// entries ids (ref-1, ref-2, ...). Pages without a References / 參考文獻 list are left alone.
const REF_HEADINGS = new Set(['references', '參考文獻']);
const SKIP = new Set(['code', 'pre', 'a', 'math', 'script', 'style', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']);
const isElement = (n) => n.type === 'element' || n.type === 'mdxJsxFlowElement' || n.type === 'mdxJsxTextElement';
const textOf = (n) => (n.type === 'text' ? n.value : (n.children ?? []).map(textOf).join(''));
const classList = (n) => {
  const c = n.properties?.className ?? n.properties?.class;
  return Array.isArray(c) ? c : typeof c === 'string' ? c.split(/\s+/) : [];
};
const CITATION = /\[(\d{1,3}(?:\s*[,–-]\s*\d{1,3})*)\]/g;

function linkify(value, count) {
  const out = [];
  let last = 0, changed = false;
  for (const m of value.matchAll(CITATION)) {
    const parts = m[1].split(/(\s*[,–-]\s*)/);
    const numbers = parts.filter((_, i) => i % 2 === 0).map(Number);
    if (numbers.some((n) => n < 1 || n > count)) continue;
    changed = true;
    if (m.index > last) out.push({ type: 'text', value: value.slice(last, m.index) });
    out.push({ type: 'text', value: '[' });
    parts.forEach((part, i) => {
      if (i % 2 === 1) out.push({ type: 'text', value: part });
      else out.push({ type: 'element', tagName: 'a', properties: { href: `#ref-${part}`, className: ['cite'] }, children: [{ type: 'text', value: part }] });
    });
    out.push({ type: 'text', value: ']' });
    last = m.index + m[0].length;
  }
  if (!changed) return null;
  if (last < value.length) out.push({ type: 'text', value: value.slice(last) });
  return out;
}

function walk(node, count) {
  if (!node.children) return;
  const next = [];
  for (const child of node.children) {
    if (child.type === 'text') {
      const parts = linkify(child.value, count);
      if (parts) next.push(...parts); else next.push(child);
      continue;
    }
    if (isElement(child) && !SKIP.has(child.tagName ?? child.name) && !classList(child).some((c) => c.startsWith('katex'))) walk(child, count);
    next.push(child);
  }
  node.children = next;
}

export default function rehypeCitations() {
  return (tree) => {
    const top = tree.children;
    const h = top.findIndex((n) => n.type === 'element' && n.tagName === 'h2' && REF_HEADINGS.has(textOf(n).trim().toLowerCase()));
    if (h < 0) return;
    const list = top.slice(h + 1).find((n) => n.type === 'element' && n.tagName === 'ol');
    if (!list) return;
    const items = list.children.filter((n) => n.type === 'element' && n.tagName === 'li');
    items.forEach((li, i) => { li.properties = { ...li.properties, id: `ref-${i + 1}` }; });
    const before = { children: top.slice(0, h) };
    walk(before, items.length);
    tree.children = [...before.children, ...top.slice(h)];
  };
}

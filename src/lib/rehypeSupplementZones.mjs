// Splits a tutorial into the main text and its "Supplementary notes" / "補充資料" section so that the notes can be shown on a
// page of their own (/<lang>/learn/<track>/<slug>/notes/). The MDX tree becomes
//   <MainZone>…main text…</MainZone> <SupplementZone items="[…]">…notes…</SupplementZone> …references…
// and the tutorial route decides, per page, which zone renders. In-page links are rewritten so that they keep working
// across the split: main text → a note becomes notes/#id, a note → the main text becomes ../#id. Citations (#ref-n) need
// no change because the reference list is shown on both pages. Runs after rehypeHeadingIds, so headings already have ids.
const NOTES = new Set(['supplementary notes', '補充資料']);
const REFS = new Set(['references', '參考文獻']);
const textOf = (n) => (n.type === 'text' ? n.value : (n.children ?? []).map(textOf).join(''));
const isH2 = (n, names) => n.type === 'element' && n.tagName === 'h2' && names.has(textOf(n).trim().toLowerCase());

function visit(node, fn) {
  fn(node);
  for (const child of node.children ?? []) visit(child, fn);
}

function rewrite(nodes, test, to) {
  for (const node of nodes) visit(node, (n) => {
    const href = n.type === 'element' && n.tagName === 'a' ? n.properties?.href : undefined;
    if (typeof href === 'string' && href.startsWith('#') && test(decodeURIComponent(href.slice(1)))) n.properties.href = to + href;
  });
}

export default function rehypeSupplementZones() {
  return (tree, file) => {
    if (!String(file.path ?? file.history?.[0] ?? '').endsWith('.mdx')) return;
    const top = tree.children;
    const s = top.findIndex((n) => isH2(n, NOTES));
    if (s < 0) return;
    let r = top.findIndex((n, i) => i > s && n.type === 'element' && n.tagName === 'h2');
    if (r < 0 || !isH2(top[r], REFS)) r = top.length;

    const esm = top.filter((n, i) => i < s && n.type === 'mdxjsEsm');
    const main = top.filter((n, i) => i < s && n.type !== 'mdxjsEsm');
    const notes = top.slice(s, r);
    const rest = top.slice(r);

    const noteIds = new Set();
    const items = [];
    for (const node of notes) visit(node, (n) => {
      const id = n.type === 'element' ? n.properties?.id : undefined;
      if (typeof id !== 'string') return;
      noteIds.add(id);
      if (n.tagName === 'h3') items.push({ id, text: textOf(n).trim() });
    });
    const heading = notes[0];
    rewrite(main, (id) => noteIds.has(id) && id !== heading.properties?.id, 'notes/');
    rewrite(notes, (id) => !noteIds.has(id) && !id.startsWith('ref-'), '../');

    const zone = (name, children, attributes = []) => ({ type: 'mdxJsxFlowElement', name, attributes, children });
    tree.children = [
      ...esm,
      zone('MainZone', main),
      zone('SupplementZone', notes, [
        { type: 'mdxJsxAttribute', name: 'items', value: JSON.stringify(items) },
        { type: 'mdxJsxAttribute', name: 'heading', value: textOf(heading).trim() },
        { type: 'mdxJsxAttribute', name: 'headingId', value: String(heading.properties?.id ?? '') },
      ]),
      ...rest,
    ];
  };
}

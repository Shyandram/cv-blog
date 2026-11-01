// Splits a tutorial into its main text and up to two optional sections, each shown on a page of its own:
//   "## Supplementary notes" / "## 補充資料"  (concept and sub-field details)  → /<lang>/learn/<track>/<slug>/notes/
//   "## Paper guides" / "## 論文導讀"          (reading guides to single papers) → /<lang>/learn/<track>/<slug>/papers/
// The MDX tree becomes
//   <MainZone>…</MainZone> <SupplementZone kind="notes" …>…</SupplementZone> <PapersZone kind="papers" …>…</PapersZone> …references…
// and the tutorial route decides, per page, which zone renders (the others become short teaser cards or nothing). In-page links
// are rewritten so that they keep working across the split (main → notes/#id or papers/#id, a note → ../#id or ../papers/#id,
// and so on). Citations (#ref-n) need no change: the reference list is shown on every page. Runs after rehypeHeadingIds.
const HEADINGS = {
  notes: new Set(['supplementary notes', '補充資料']),
  papers: new Set(['paper guides', '論文導讀']),
  rest: new Set(['references', '參考文獻']),
};
const PREFIX = {
  main: { notes: 'notes/', papers: 'papers/' },
  notes: { main: '../', papers: '../papers/' },
  papers: { main: '../', notes: '../notes/' },
};
const textOf = (n) => (n.type === 'text' ? n.value : (n.children ?? []).map(textOf).join(''));

function visit(node, fn) {
  fn(node);
  for (const child of node.children ?? []) visit(child, fn);
}

function zoneOf(node, current) {
  if (node.type !== 'element' || node.tagName !== 'h2') return current;
  const text = textOf(node).trim().toLowerCase();
  for (const [zone, names] of Object.entries(HEADINGS)) if (names.has(text)) return zone;
  return current === 'rest' ? 'rest' : 'main';
}

export default function rehypeSupplementZones() {
  return (tree, file) => {
    if (!String(file.path ?? file.history?.[0] ?? '').endsWith('.mdx')) return;
    const top = tree.children;
    if (!top.some((n) => ['notes', 'papers'].includes(zoneOf(n, 'main')))) return;

    const zones = { main: [], notes: [], papers: [], rest: [] };
    const esm = [];
    let current = 'main';
    for (const node of top) {
      if (node.type === 'mdxjsEsm') { esm.push(node); continue; }
      current = zoneOf(node, current);
      zones[current].push(node);
    }

    // Which zone owns each heading/element id.
    const owner = new Map();
    for (const zone of ['main', 'notes', 'papers']) for (const node of zones[zone]) visit(node, (n) => {
      const id = n.type === 'element' ? n.properties?.id : undefined;
      if (typeof id === 'string') owner.set(id, zone);
    });
    for (const zone of ['main', 'notes', 'papers']) for (const node of zones[zone]) visit(node, (n) => {
      const href = n.type === 'element' && n.tagName === 'a' ? n.properties?.href : undefined;
      if (typeof href !== 'string' || !href.startsWith('#')) return;
      const target = owner.get(decodeURIComponent(href.slice(1)));
      const prefix = target && target !== zone ? PREFIX[zone][target] : undefined;
      // A link to a section's own h2 from the main text should land on its teaser card, which keeps that id.
      if (prefix && !(zone === 'main' && zones[target][0]?.properties?.id === decodeURIComponent(href.slice(1)))) n.properties.href = prefix + href;
    });

    const attr = (name, value) => ({ type: 'mdxJsxAttribute', name, value });
    const wrap = (name, kind) => {
      const nodes = zones[kind];
      const items = [];
      for (const node of nodes) visit(node, (n) => {
        if (n.type === 'element' && n.tagName === 'h3' && typeof n.properties?.id === 'string') items.push({ id: n.properties.id, text: textOf(n).trim() });
      });
      const heading = nodes[0];
      return { type: 'mdxJsxFlowElement', name, children: nodes, attributes: [
        attr('kind', kind), attr('items', JSON.stringify(items)), attr('heading', textOf(heading).trim()), attr('headingId', String(heading.properties?.id ?? '')),
      ] };
    };
    const ordered = ['notes', 'papers'].filter((k) => zones[k].length > 0)
      .sort((a, b) => top.indexOf(zones[a][0]) - top.indexOf(zones[b][0]))
      .map((k) => wrap(k === 'notes' ? 'SupplementZone' : 'PapersZone', k));
    tree.children = [...esm, { type: 'mdxJsxFlowElement', name: 'MainZone', attributes: [], children: zones.main }, ...ordered, ...zones.rest];
  };
}

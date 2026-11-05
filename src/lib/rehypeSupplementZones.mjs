import { chapterPath } from './chapter-paths.mjs';
// Splits a tutorial into its main text and up to two optional sections, and each item of those sections into a page of its own
// (a "sub-chapter"):
//   "## Supplementary notes" / "## 補充資料"  → index /<lang>/learn/<track>/<slug>/notes/,  one page per ### at …/notes/<h3-id>/
//   "## Paper guides" / "## 論文導讀"          → index /<lang>/learn/<track>/<slug>/papers/, one page per ### at …/papers/<h3-id>/
// The MDX tree becomes
//   <MainZone>…</MainZone>
//   <SupplementZone kind="notes" items heading headingId ids>
//     <ZoneIntro>…text between the ## and the first ###…</ZoneIntro>
//     <ZoneItems><ZoneItem kind id title n><ItemLead>first sentence</ItemLead><ItemBody>…the item without its ###…</ItemBody></ZoneItem>…</ZoneItems>
//   </SupplementZone>
//   <PapersZone kind="papers" …>…same shape…</PapersZone>
//   <RestZone>…references…</RestZone>
// and the tutorial route maps these names to components per page (src/components/zones/), so that each page renders only its
// part. The section's ## and each item's ### are not rendered inside the content: the page's h1 carries their text and id.
// In-page links are rewritten to the relative URL of the page that holds their target (main → notes/<item>/#id, an item →
// ../../#id, ../<other>/#id, ../../papers/<item>/#id, …); a link to an item's own ### drops the hash. `ids` maps every id of a
// section to its item so that old links (…/notes/#id, …/#note-id) can be forwarded in the browser. Citations (#ref-n) need no
// change: the reference list is shown on every page. Runs after rehypeHeadingIds (which has already collected the headings).
const HEADINGS = {
  notes: new Set(['supplementary notes', '補充資料']),
  papers: new Set(['paper guides', '論文導讀']),
  rest: new Set(['references', '參考文獻']),
};
const textOf = (n) => (n.type === 'text' ? n.value : (n.children ?? []).map(textOf).join(''));
const isEl = (n, tag) => n?.type === 'element' && (!tag || n.tagName === tag);
const isKatex = (n) => isEl(n) && Array.isArray(n.properties?.className) && n.properties.className.includes('katex');

function visit(node, fn) {
  fn(node);
  for (const child of node.children ?? []) visit(child, fn);
}

function zoneOf(node, current) {
  if (!isEl(node, 'h2')) return current;
  const text = textOf(node).trim().toLowerCase();
  for (const [zone, names] of Object.entries(HEADINGS)) if (names.has(text)) return zone;
  return current === 'rest' ? 'rest' : 'main';
}

// Relative URL from one page of the chapter to another; both are directory paths relative to the chapter ('' is the main page).
function relPath(from, to) {
  const a = from.split('/').filter(Boolean);
  const b = to.split('/').filter(Boolean);
  let k = 0;
  while (k < a.length && k < b.length && a[k] === b[k]) k++;
  return '../'.repeat(a.length - k) + b.slice(k).map((s) => `${s}/`).join('');
}

// The first sentence of an item (the house style opens every item with one sentence saying what it is about): a copy of its first
// paragraph cut after the first sentence end that is not inside math, an abbreviation or an initial; ids are dropped so they
// stay unique.
const SENTENCE_END = /(?<!\b(?:e\.g|i\.e|al|vs|cf|Fig|Figs|Eq|Eqs|resp|approx|No|Sec|ca|[A-Z]))[.!?](?=\s|$)|[。！？]/;
function firstSentence(nodes) {
  let para;
  for (const n of nodes) { visit(n, (x) => { if (!para && isEl(x, 'p')) para = x; }); if (para) break; }
  if (!para) return [];
  const copy = structuredClone(para);
  visit(copy, (x) => { if (x.properties) delete x.properties.id; });
  const cut = (node) => {
    if (node.type === 'text') {
      const m = SENTENCE_END.exec(node.value);
      if (!m) return false;
      node.value = node.value.slice(0, m.index + m[0].length);
      return true;
    }
    if (isKatex(node)) return false;
    const kids = node.children ?? [];
    for (let i = 0; i < kids.length; i++) {
      // A bold run-in label that opens the paragraph ("**Problem.** …") is kept whole; the sentence after it is the lead.
      if (node === copy && i === 0 && isEl(kids[i], 'strong')) continue;
      if (cut(kids[i])) { node.children = kids.slice(0, i + 1); return true; }
    }
    return false;
  };
  cut(copy);
  return [copy];
}

export default function rehypeSupplementZones() {
  return (tree, file) => {
    if (!String(file.path ?? file.history?.[0] ?? '').endsWith('.mdx')) return;
    visit(tree, (n) => { if (isEl(n, 'a') && typeof n.properties?.href === 'string') n.properties.href = chapterPath(n.properties.href); });
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

    // Each optional section: its ## heading, the intro before the first ###, and one item per ###.
    const sections = {};
    for (const kind of ['notes', 'papers']) {
      if (!zones[kind].length) continue;
      const [heading, ...nodes] = zones[kind];
      const s = { heading, intro: [], items: [] };
      for (const node of nodes) {
        if (isEl(node, 'h3') && typeof node.properties?.id === 'string') s.items.push({ id: node.properties.id, title: textOf(node).trim(), h3: node, body: [] });
        else if (s.items.length) s.items.at(-1).body.push(node);
        else s.intro.push(node);
      }
      sections[kind] = s;
    }

    // The page (directory relative to the chapter) that renders each node group, and the owner of every id.
    const owner = new Map();
    const own = (nodes, page) => { for (const node of nodes) visit(node, (n) => { if (isEl(n) && typeof n.properties?.id === 'string') owner.set(n.properties.id, page); }); };
    own(zones.main, '');
    const groups = [{ nodes: zones.main, page: '' }];
    for (const [kind, s] of Object.entries(sections)) {
      own([s.heading, ...s.intro], `${kind}/`);
      groups.push({ nodes: s.intro, page: `${kind}/` });
      for (const item of s.items) {
        own([item.h3, ...item.body], `${kind}/${item.id}/`);
        groups.push({ nodes: item.body, page: `${kind}/${item.id}/` });
      }
    }
    const pageHeads = new Set(Object.values(sections).flatMap((s) => s.items.map((x) => x.id)));
    const rewrite = (nodes, page) => { for (const node of nodes) visit(node, (n) => {
      const href = isEl(n, 'a') ? n.properties?.href : undefined;
      if (typeof href !== 'string' || !href.startsWith('#')) return;
      const id = decodeURIComponent(href.slice(1));
      const target = owner.get(id);
      if (target === undefined || target === page) return;
      // A link to a section's own ## from the main text lands on its teaser card, which keeps that id.
      if (page === '' && Object.values(sections).some((s) => s.heading.properties?.id === id)) return;
      n.properties.href = relPath(page, target) + (pageHeads.has(id) ? '' : href);
    }); };
    for (const g of groups) rewrite(g.nodes, g.page);

    const attr = (name, value) => ({ type: 'mdxJsxAttribute', name, value });
    const el = (name, children, attributes = []) => ({ type: 'mdxJsxFlowElement', name, attributes, children });
    const wrap = (kind) => {
      const s = sections[kind];
      const ids = {};
      for (const item of s.items) for (const node of [item.h3, ...item.body]) visit(node, (n) => { if (isEl(n) && typeof n.properties?.id === 'string') ids[n.properties.id] = item.id; });
      const items = s.items.map((item, i) => {
        const lead = firstSentence(item.body);
        rewrite(lead, `${kind}/`);
        return el('ZoneItem', [el('ItemLead', lead), el('ItemBody', item.body)],
          [attr('kind', kind), attr('id', item.id), attr('title', item.title), attr('n', String(i + 1))]);
      });
      return el(kind === 'notes' ? 'SupplementZone' : 'PapersZone', [el('ZoneIntro', s.intro), el('ZoneItems', items)], [
        attr('kind', kind), attr('items', JSON.stringify(s.items.map((x) => ({ id: x.id, text: x.title })))), attr('heading', textOf(s.heading).trim()),
        attr('headingId', String(s.heading.properties?.id ?? '')), attr('ids', JSON.stringify(ids)),
      ]);
    };
    const ordered = Object.keys(sections).sort((a, b) => top.indexOf(zones[a][0]) - top.indexOf(zones[b][0])).map(wrap);
    tree.children = [...esm, el('MainZone', zones.main), ...ordered, ...(zones.rest.length ? [el('RestZone', zones.rest)] : [])];
  };
}

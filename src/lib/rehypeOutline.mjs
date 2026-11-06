// Leveled outlines (AGENTS.md §3 "Showing structure and layers").
//
// Every nested Markdown list in prose already gets level styling from global.css (indent, connector line, smaller and
// muted third level). This plugin adds the opt-in part: a list item may start with a priority marker,
//
//   - [core] **Precision** answers "when the model says positive, how often is it right?"
//     - [key] It matters when a false alarm is costly.
//       - [detail] A spam filter that hides real mail.
//
// The marker becomes a text tag ("Core" / "核心", "Important" / "重要", "Detail" / "細節") whose shape also differs
// (solid, outlined, dashed), so priority never depends on colour. A list with at least one marker is an outline: its
// first level is set as key points, and a long third level (3+ items or 240+ characters, a CJK character counting 2.5) folds into a native
// <details> ("2 more details"), closed in the HTML, so it works without JavaScript and prints open (src/scripts/prose.ts).
// The markers are the same English words in both languages; the PDF exporter (scripts/book/export.mjs) turns them into
// a bold label.
const MARK = /^\s*\[(core|key|detail)\]\s*/;
const LABEL = {
  en: { core: 'Core', key: 'Important', detail: 'Detail' },
  zh: { core: '核心', key: '重要', detail: '細節' },
};
// Reading weight of a text: a CJK character carries about as much as 2.5 Latin characters.
const weight = (t) => t.trim().length + 1.5 * (t.match(/[\u3000-\u9fff\uff00-\uffef]/g) ?? []).length;
const isList = (n) => n?.type === 'element' && (n.tagName === 'ul' || n.tagName === 'ol');
const isItem = (n) => n?.type === 'element' && n.tagName === 'li';
const textOf = (n) => (n.type === 'text' ? n.value : (n.children ?? []).map(textOf).join(''));
const element = (tagName, properties, children) => ({ type: 'element', tagName, properties, children });
const text = (value) => ({ type: 'text', value });
const addClass = (n, c) => {
  n.properties ??= {};
  const cur = n.properties.className ?? [];
  n.properties.className = [...(Array.isArray(cur) ? cur : String(cur).split(/\s+/)), c].filter(Boolean);
};

/** First text node of an item, looking through a leading <p> (loose lists) only. */
function leadText(li) {
  let node = li.children.find((c) => !(c.type === 'text' && !c.value.trim()));
  if (node?.type === 'element' && node.tagName === 'p') node = node.children[0];
  return node?.type === 'text' ? node : undefined;
}

export default function rehypeOutline() {
  return (tree, file) => {
    const zh = /[\\/]zh-tw[\\/]/.test(String(file?.path ?? file?.history?.[0] ?? ''));
    const L = zh ? LABEL.zh : LABEL.en;
    // Pass 1: turn markers into tags; remember which top-level lists contain one.
    const mark = (list) => {
      let found = false;
      for (const li of list.children.filter(isItem)) {
        const lead = leadText(li);
        const m = lead && lead.value.match(MARK);
        if (m) {
          found = true;
          lead.value = lead.value.slice(m[0].length);
          addClass(li, `pri-item`);
          li.properties.dataPriority = m[1];
          const host = li.children[0]?.type === 'element' && li.children[0].tagName === 'p' ? li.children[0] : li;
          host.children.splice(host.children.indexOf(lead), 0,
            element('span', { className: ['pri', `pri-${m[1]}`], dataPagefindIgnore: '' }, [
              text(L[m[1]]), element('span', { className: ['sr-only'] }, [text(zh ? '：' : ': ')]),
            ]));
        }
        for (const child of li.children) if (isList(child) && mark(child)) found = true;
      }
      return found;
    };
    // Pass 2: an outline folds its long third level.
    const fold = (list, depth) => {
      for (const li of list.children.filter(isItem)) {
        li.children = li.children.map((child) => {
          if (!isList(child)) return child;
          fold(child, depth + 1);
          if (depth + 1 !== 3) return child;
          const items = child.children.filter(isItem);
          const long = items.length >= 3 || weight(textOf(child)) >= 240;
          if (!long) return child;
          const n = items.length;
          const label = zh ? `另有 ${n} 項細節` : `${n} more detail${n === 1 ? '' : 's'}`;
          return element('details', { className: ['outline-more'] }, [
            element('summary', { dataPagefindIgnore: '' }, [text(label)]),
            child,
          ]);
        });
      }
    };
    const walk = (node, inList) => {
      for (const child of node.children ?? []) {
        if (isList(child) && !inList) {
          if (mark(child)) { addClass(child, 'outline'); fold(child, 1); }
          walk(child, true);
        } else walk(child, inList || isList(child));
      }
    };
    walk(tree, false);
  };
}

import GithubSlugger from 'github-slugger';
import legacy from '../content/paper-heading-ids.json' with {type: 'json'};

const textOf = n => n.type === 'text' ? n.value : (n.children ?? []).map(textOf).join('');
const normalize = s => s.replace(/\s+/g, ' ').trim();
const byFile = new Map();
for (const heading of legacy.headings) {
  if (!byFile.has(heading.file)) byFile.set(heading.file, []);
  byFile.get(heading.file).push(heading);
}

// Paper guides are independent pages. Chapter-wide duplicate counters must not
// change their bookmarks when another guide is inserted or reordered.
// Run before Astro's heading collector so routes, TOCs and rewritten links agree.
export default function rehypePaperHeadingIds() {
  return (tree, file) => {
    const filename = String(file.path ?? file.history?.[0] ?? '').replaceAll('\\', '/');
    const match = filename.match(/(?:^|\/)src\/content\/tutorials\/(.+\.mdx)$/);
    if (!match) return;
    const retained = byFile.get(match[1]) ?? [];
    const guideNames = new GithubSlugger();
    let inPapers = false, guide = '', prefix = '', local;
    const consumed = new Set();
    const visit = node => {
      if (node.type === 'element' && /^h[2-6]$/.test(node.tagName)) {
        const depth = Number(node.tagName[1]), title = normalize(textOf(node));
        if (depth === 2) { inPapers = /^(Paper guides|論文導讀)$/.test(title); guide = ''; }
        else if (inPapers && depth === 3) {
          guide = title; prefix = `paper-${guideNames.slug(title)}--`; local = new GithubSlugger();
        } else if (inPapers && guide && depth >= 4) {
          node.properties ??= {};
          if (typeof node.properties.id !== 'string') {
            const previous = retained.find(h => h.guide === guide && h.depth === depth && h.text === title && !consumed.has(h));
            const suffix = local.slug(title);
            node.properties.id = previous ? previous.id : prefix+suffix;
            if (previous) consumed.add(previous);
          }
        }
      }
      for (const child of node.children ?? []) visit(child);
    };
    visit(tree);
  };
}

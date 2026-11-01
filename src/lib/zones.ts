// The optional sections of a tutorial and their items, from its rendered headings (so the ids are exactly the ones
// rehypeHeadingIds gave them): "## Supplementary notes" / "## 補充資料" and "## Paper guides" / "## 論文導讀", one item per
// "###" (with the "####"s inside it). Every item is a page of its own (src/lib/rehypeSupplementZones.mjs):
// …/<slug>/notes/<item-id>/ and …/<slug>/papers/<item-id>/, with an index page at …/<slug>/notes/ and …/<slug>/papers/.
export type ZoneKind = 'notes' | 'papers';
export type ZoneHeading = { slug: string; text: string };
export type ZoneItem = ZoneHeading & { subs: ZoneHeading[] };
export type Zone = { heading: ZoneHeading; items: ZoneItem[] };
export type Zones = Partial<Record<ZoneKind, Zone>>;

export const ZONE_KINDS: ZoneKind[] = ['notes', 'papers'];
const SECTION: Record<ZoneKind, RegExp> = { notes: /^(supplementary notes|補充資料)$/i, papers: /^(paper guides|論文導讀)$/i };

export function zonesOf(headings: { depth: number; slug: string; text: string }[]): Zones {
  const out: Zones = {};
  let cur: ZoneKind | undefined;
  for (const h of headings) {
    if (h.depth === 2) {
      cur = ZONE_KINDS.find((k) => SECTION[k].test(h.text.trim()));
      if (cur) out[cur] = { heading: { slug: h.slug, text: h.text }, items: [] };
    } else if (cur && h.depth === 3) out[cur]!.items.push({ slug: h.slug, text: h.text, subs: [] });
    else if (cur && h.depth === 4) out[cur]!.items.at(-1)?.subs.push({ slug: h.slug, text: h.text });
  }
  return out;
}

/** UI names of the two sections. */
export const ZONE_LABEL: Record<ZoneKind, { en: string; 'zh-tw': string }> = {
  notes: { en: 'Supplementary notes', 'zh-tw': '補充資料' },
  papers: { en: 'Paper guides', 'zh-tw': '論文導讀' },
};

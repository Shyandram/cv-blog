// Native disclosures are present in the HTML, so code starts folded without a flash
// and remains available without JavaScript. Keep the original pre/code tree intact.
const textOf = (node) => node.type === 'text' ? node.value : (node.children ?? []).map(textOf).join('');
const element = (tagName, properties, children) => ({ type: 'element', tagName, properties, children });
const text = (value) => ({ type: 'text', value });

export default function rehypeCodeFolding() {
  return (tree, file) => {
    const zh = /[\\/]zh-tw[\\/]/.test(String(file?.path ?? file?.history?.[0] ?? ''));
    const usedIds = new Set();
    const collect = (node) => {
      if (node.properties?.id) usedIds.add(node.properties.id);
      (node.children ?? []).forEach(collect);
    };
    collect(tree);
    let number = 0;
    const walk = (node) => {
      if (!node.children || node.properties?.className?.includes('rt-code')) return;
      node.children = node.children.map((child) => {
        const code = child.tagName === 'pre' && child.children?.find((n) => n.tagName === 'code');
        if (!code) { walk(child); return child; }
        const classes = code.properties?.className ?? [];
        const language = child.properties?.dataLanguage ?? classes.find?.((c) => c.startsWith('language-'))?.slice(9);
        const names = { python: 'Python', py: 'Python', javascript: 'JavaScript', js: 'JavaScript', typescript: 'TypeScript', ts: 'TypeScript', bash: 'Bash', sh: 'Shell', json: 'JSON', html: 'HTML', css: 'CSS' };
        const languageLabel = language && !['text', 'plaintext', 'plain'].includes(language) ? (names[language] ?? language) : '';
        const lines = textOf(code).replace(/\n$/, '').split('\n').length;
        child.properties ??= {};
        if (!child.properties.id) {
          let id;
          do { id = `code-example-${++number}`; } while (usedIds.has(id));
          child.properties.id = id;
          usedIds.add(id);
        }
        const label = [zh ? '程式碼' : 'Code', languageLabel, zh ? `${lines} 行` : `${lines} ${lines === 1 ? 'line' : 'lines'}`].filter(Boolean).join(' · ');
        const summary = element('summary', { dataPagefindIgnore: '' }, [
          element('span', { className: ['code-label'] }, [text(label)]),
          element('span', { className: ['code-action'], ariaHidden: 'true' }, [
            element('span', { className: ['code-closed'] }, [text(zh ? '展開' : 'Expand')]),
            element('span', { className: ['code-open'] }, [text(zh ? '收合' : 'Collapse')]),
          ]),
        ]);
        return element('div', { className: ['rt-code'] }, [
          element('details', { className: ['code-fold'] }, [summary, child]),
        ]);
      });
    };
    walk(tree);
  };
}

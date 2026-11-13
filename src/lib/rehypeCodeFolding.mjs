// Native disclosures are present in the HTML, so code starts folded without a flash
// and remains available without JavaScript. Keep the original pre/code tree intact.
// Tutorials only: a plain-text block that directly follows a code block (nothing but whitespace between them) is that
// code's printed output. It is attached below the code in the same frame (div.rt-output) and stays visible while the
// code is folded.
const textOf = (node) => node.type === 'text' ? node.value : (node.children ?? []).map(textOf).join('');
const element = (tagName, properties, children) => ({ type: 'element', tagName, properties, children });
const text = (value) => ({ type: 'text', value });
const OUTPUT_LANGUAGES = ['text', 'plaintext', 'txt'];

export default function rehypeCodeFolding() {
  return (tree, file) => {
    const path = String(file?.path ?? file?.history?.[0] ?? '');
    const zh = /[\\/]zh-tw[\\/]/.test(path);
    const tutorial = /[\\/]tutorials[\\/]/.test(path);
    const usedIds = new Set();
    const collect = (node) => {
      if (node.properties?.id) usedIds.add(node.properties.id);
      (node.children ?? []).forEach(collect);
    };
    collect(tree);
    let number = 0;
    const languageOf = (pre) => {
      const classes = pre.children?.find((n) => n.tagName === 'code')?.properties?.className ?? [];
      return pre.properties?.dataLanguage ?? classes.find?.((c) => c.startsWith('language-'))?.slice(9);
    };
    const isCode = (node) => node?.tagName === 'pre' && node.children?.some((n) => n.tagName === 'code');
    const walk = (node) => {
      if (!node.children || node.properties?.className?.includes('rt-code')) return;
      // Pair code blocks with the output block right after them (tutorials only).
      const outputs = new Set();
      if (tutorial) {
        let prev;
        for (const child of node.children) {
          if (child.type === 'text' && !child.value.trim()) continue;
          if (isCode(child) && isCode(prev) && !OUTPUT_LANGUAGES.includes(languageOf(prev)) && OUTPUT_LANGUAGES.includes(languageOf(child))) {
            outputs.add(child);
            prev = undefined;
            continue;
          }
          prev = child;
        }
      }
      const out = [];
      let lastWrap;
      for (const child of node.children) {
        if (outputs.has(child)) {
          // Drop the whitespace between the code and its output.
          while (out.length && out[out.length - 1] !== lastWrap) out.pop();
          lastWrap.children.push(element('div', { className: ['rt-output'] }, [
            element('p', { className: ['rt-output-label'], dataPagefindIgnore: '' }, [text(zh ? '輸出' : 'Output')]),
            child,
          ]));
          continue;
        }
        if (!isCode(child)) { walk(child); out.push(child); continue; }
        const code = child.children.find((n) => n.tagName === 'code');
        const language = languageOf(child);
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
        lastWrap = element('div', { className: ['rt-code'] }, [
          element('details', { className: ['code-fold'] }, [summary, child]),
        ]);
        out.push(lastWrap);
      }
      node.children = out;
    };
    walk(tree);
  };
}

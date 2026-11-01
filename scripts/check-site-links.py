"""Check every built internal HTML link, including fragments and legacy forwarding."""
import json
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote, urljoin

class Page(HTMLParser):
    def __init__(self, source):
        super().__init__(); self.ids=set(); self.links=[]; self.forward=set(); self.redirect=None
        self.feed(source)
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if a.get('id'): self.ids.add(a['id'])
        if tag=='a' and 'href' in a: self.links.append(a['href'])
        if a.get('data-zone-ids'):
            self.forward.update(json.loads(a['data-zone-ids']))
        if tag=='meta' and a.get('http-equiv','').lower()=='refresh':
            self.redirect=a.get('content','').split('url=',1)[-1].strip()

root=Path('dist'); pages={}
for f in root.rglob('*.html'):
    key='/'+f.relative_to(root).as_posix()
    if key.endswith('/index.html'): key=key[:-10]
    pages[key]=Page(f.read_text())
errors=[]; checked=0
for origin,page in pages.items():
    for href in page.links:
        u=urlsplit(urljoin('https://blog.shyandram.dev'+origin,href))
        if u.netloc!='blog.shyandram.dev': continue
        path=unquote(u.path); fragment=unquote(u.fragment); target=pages.get(path)
        if not target:
            if (root/path.lstrip('/')).is_file(): continue
            errors.append((origin,href,'missing page')); continue
        checked+=1
        for _ in range(4):
            if not target.redirect: break
            dest=urlsplit(urljoin('https://blog.shyandram.dev'+path,target.redirect))
            path=unquote(dest.path); target=pages.get(path)
            if not target: break
        if target and fragment and fragment not in target.ids and fragment not in target.forward:
            errors.append((origin,href,'missing fragment'))
print(f'{len(pages)} pages; {checked} links; {len(errors)} broken')
Path('artifacts/link-errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2))
for row in errors[:20]:print(*row)
raise SystemExit(bool(errors))

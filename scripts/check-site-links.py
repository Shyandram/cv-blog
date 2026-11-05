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
assert len(pages) > 100, 'No complete site build found; run npm run build first'
def resolve(path, fragment=''):
    seen=set()
    while path in pages:
        if path in seen:
            return None, fragment, 'redirect loop'
        seen.add(path)
        target=pages[path]
        if not target.redirect:
            return target, fragment, None
        dest=urlsplit(urljoin('https://blog.shyandram.dev'+path,target.redirect))
        if dest.fragment:
            fragment=unquote(dest.fragment)
        path=unquote(dest.path)
    return None, fragment, 'missing redirect target'

errors=[]; checked=0
# Check redirects themselves too: unlinked self redirects can otherwise look valid.
for origin,page in pages.items():
    if page.redirect:
        _,_,error=resolve(origin)
        if error:
            errors.append((origin,page.redirect,error))
    for href in page.links:
        u=urlsplit(urljoin('https://blog.shyandram.dev'+origin,href))
        if u.netloc!='blog.shyandram.dev': continue
        path=unquote(u.path); fragment=unquote(u.fragment)
        if path not in pages:
            if (root/path.lstrip('/')).is_file(): continue
            errors.append((origin,href,'missing page')); continue
        checked+=1
        target,fragment,error=resolve(path,fragment)
        if error:
            errors.append((origin,href,error))
        elif fragment and fragment not in target.ids and fragment not in target.forward:
            errors.append((origin,href,'missing fragment'))
print(f'{len(pages)} pages; {checked} links; {len(errors)} broken')
Path('artifacts/link-errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2))
for row in errors[:20]:print(*row)
raise SystemExit(bool(errors))

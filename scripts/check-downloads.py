"""Check bilingual download layouts, edition metadata and all built PDF hashes."""
from pathlib import Path
import hashlib,json
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];expected=json.loads((R/'src/data/books.json').read_text())['sourceCommit'];base='http://127.0.0.1:4322';results=[]
with sync_playwright() as p:
 b=p.chromium.launch(args=['--no-sandbox'])
 for lang in ['en','zh-tw']:
  for width in [1440,375]:
   page=b.new_page(viewport={'width':width,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
   assert page.goto(base+f'/{lang}/download/').status==200
   page.evaluate('document.fonts.ready')
   assert page.locator('.book').count()==2
   assert page.locator('.volumes > li').count()==8
   assert page.locator('.book .meta code').all_text_contents()==[expected,expected]
   assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
   assert 'online' in page.locator('.prose').inner_text().lower() if lang=='en' else '不收進 PDF' in page.locator('.prose').inner_text()
   for a in page.locator('a[download]').all():
    path=a.get_attribute('href');built=R/'dist'/path.lstrip('/');source=R/'public'/path.lstrip('/')
    assert built.exists() and hashlib.sha256(built.read_bytes()).digest()==hashlib.sha256(source.read_bytes()).digest()
   assert not errors,errors
   page.screenshot(path=str(R/'artifacts'/f'{lang}-{width}-final-downloads.png'))
   results.append({'lang':lang,'width':width,'pdfs':8,'sourceCommit':expected});page.close()
 b.close()
(R/'docs/worklog/paper-guide-download-validation.json').write_text(json.dumps(results,indent=2)+'\n');print('Four download layouts, eight PDF copies and source metadata passed.')

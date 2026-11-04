"""Check the rewritten bilingual learning path and optional-page language switching."""
import json
from pathlib import Path
from urllib.parse import unquote
from reader_test_helpers import set_reader_group
from playwright.sync_api import sync_playwright
BASE='http://127.0.0.1:4322'
slugs={'dl-03a-real-time-detection','dl-04a-super-resolution','dl-04d-blind-real-world-super-resolution','dl-04e-one-step-diffusion-super-resolution','dl-05-low-light-enhancement','dl-05b-image-fusion','dl-10b-one-step-generation'}
chapters=[t for t in json.loads(Path('docs/content-index.json').read_text())['tutorials'] if t['slug'].split('/')[-1] in slugs]
report=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
 for lang in ['en','zh-tw']:
  for width in [1440,375]:
   page=browser.new_page(viewport={'width':width,'height':900});errors=[]
   page.on('pageerror',lambda e:errors.append(str(e)))
   for t in chapters:
    main=t['pages']['main'].replace('/en/','/'+lang+'/')
    items=[s for s in t['sections'] if s['zone']=='papers' and s['depth']==3]
    paper=(items or [s for s in t['sections'] if s['zone']=='notes' and s['depth']==3])[0]
    optional=paper['url'] if lang=='en' else paper['urlZh']
    for route in [main,optional]:
     assert page.goto(BASE+route).status==200,route
     print('CHECK',lang,width,route,flush=True)
     page.wait_for_timeout(150)
     assert page.locator('h1').count()==1,route
     assert not page.evaluate('document.documentElement.scrollWidth>innerWidth'),route
     other='zh-tw' if lang=='en' else 'en'
     alternate=page.locator(f'header a[hreflang="{other}"]').get_attribute('href')
     expected=main.replace('/'+lang+'/','/'+other+'/') if route==main else (paper['urlZh'] if lang=='en' else paper['url'])
     assert unquote(alternate)==unquote(expected),(route,alternate,expected)
     if page.locator('.chapter-section').count():
      set_reader_group(page, 'sections', False)
      assert page.locator('.chapter-section[open]').count()==0,route
      set_reader_group(page, 'sections', True)
      assert page.locator('.chapter-section:not([open])').count()==0,route
     else:
      assert route != main,route
     assert not errors,(route,errors)
     report.append({'lang':lang,'width':width,'route':route})
    if width==375 and 'dl-03a-' in t['slug']:
     page.goto(BASE+main);page.screenshot(path=f'artifacts/{lang}-375-updated-detection.png')
   page.close()
 browser.close()
Path('artifacts/updated-chapters.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(len(report),'updated chapter/paper layouts and language switches passed')

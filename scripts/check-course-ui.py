"""Smoke test the bilingual course's navigation and mobile layout against a local build."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
BASE='http://127.0.0.1:4322'
report=[]
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox'])
 for lang in ['en','zh-tw']:
  for width in [1440,375]:
   page=b.new_page(viewport={'width':width,'height':900},device_scale_factor=1)
   errors=[]
   page.on('pageerror',lambda e:errors.append(str(e)))
   def visit(path,name):
    r=page.goto(BASE+path); assert r.status==200,(path,r.status)
    page.wait_for_timeout(200)
    assert not page.evaluate('document.documentElement.scrollWidth > innerWidth'),path
    page.screenshot(path=f'artifacts/{lang}-{width}-{name}.png')
    report.append({'path':path,'width':width,'errors':list(errors)})
   visit(f'/{lang}/learn/','learn')
   button=page.locator('.lt-toggle').first
   button.click(); assert button.get_attribute('aria-expanded')=='true'
   controlled=button.get_attribute('aria-controls')
   assert page.locator('#'+controlled).is_visible()
   visit(f'/{lang}/learn/tree/','tree')
   visit(f'/{lang}/learn/deep-learning/image-classification/','chapter')
   if width==375:
    page.locator('.rt-mnav > summary').click()
    assert page.locator('.rt-chapters').is_visible()
    page.screenshot(path=f'artifacts/{lang}-{width}-contents.png')
   notes=page.locator('.chapter-tabs a').nth(1).get_attribute('href')
   visit(notes,'notes')
   item=page.locator('.item-card > a').first.get_attribute('href')
   visit(notes+item,'note')
   assert page.locator('h1').count()==1
   papers=page.locator('.chapter-tabs a').nth(2).get_attribute('href')
   visit(papers,'papers')
   paper=page.locator('.item-card > a').first.get_attribute('href')
   visit(papers+paper,'paper')
   assert not errors,errors
   page.close()
 b.close()
Path('artifacts/ui-checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(f'{len(report)} page layouts passed; tree controls and mobile menus passed')

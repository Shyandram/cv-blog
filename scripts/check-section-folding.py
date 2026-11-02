"""Check folding, keyboard controls, anchor navigation and printing on both tutorial languages."""
from pathlib import Path
from playwright.sync_api import sync_playwright
BASE='http://127.0.0.1:4322'
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox'])
 for lang in ['en','zh-tw']:
  for width in [1440,375]:
   page=b.new_page(viewport={'width':width,'height':900})
   errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
   path=f'/{lang}/learn/deep-learning/image-classification/'
   assert page.goto(BASE+path).status==200
   page.wait_for_selector('.chapter-section')
   sections=page.locator('.chapter-section');count=sections.count()
   assert count>10,count
   assert page.locator('.chapter-section:not([open])').count()==0
   first=sections.first;summary=first.locator(':scope > summary')
   summary.click();assert first.get_attribute('open') is None
   summary.focus();summary.press('Enter');assert first.get_attribute('open') is not None
   page.locator('.section-controls button').nth(1).click()
   assert page.locator('.chapter-section[open]').count()==0
   target=page.locator('.section-body h3[id]').first.get_attribute('id')
   page.evaluate('(id)=>{location.hash=id}',target)
   page.wait_for_function('(id)=>document.getElementById(id).closest("details").open',arg=target)
   assert page.locator('.notes-teaser').first.is_visible()
   page.locator('.section-controls button').nth(1).click()
   page.evaluate('dispatchEvent(new Event("beforeprint"))')
   assert page.locator('.chapter-section[open]').count()==count
   page.evaluate('dispatchEvent(new Event("afterprint"))')
   assert page.locator('.chapter-section[open]').count()==0
   page.locator('.section-controls button').first.click()
   assert page.locator('.chapter-section[open]').count()==count
   assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
   assert not errors,errors
   page.locator('.section-controls button').nth(1).click()
   page.screenshot(path=f'artifacts/{lang}-{width}-folded.png')
   page.close()
  page=b.new_page(java_script_enabled=False)
  page.goto(BASE+f'/{lang}/learn/deep-learning/image-classification/')
  assert page.locator('.prose h2').count()>10
  assert page.locator('.section-controls').count()==0
  page.close()
 b.close()
print('4 folding layouts, keyboard, deep links, print restore and no-JS reading passed')

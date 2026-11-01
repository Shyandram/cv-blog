import json
from urllib.parse import unquote
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox'])
 for lang in ['en','zh-tw']:
  page=b.new_page()
  base=f'/{lang}/learn/deep-learning/image-classification/'
  page.goto('http://127.0.0.1:4322'+base)
  ids=json.loads(page.locator('[data-zone-ids]').first.get_attribute('data-zone-ids'))
  key=next(iter(ids));item=ids[key]
  for path in [base+'#'+key,base+'notes/#'+key,base.replace('image-classification','dl-02-image-classification')+'notes/#'+key]:
   page.goto('http://127.0.0.1:4322'+path)
   page.wait_for_url(lambda u: '/notes/'+item+'/' in unquote(u),timeout=10000)
   print('Forward passed:',lang,path,flush=True)
  page.goto('http://127.0.0.1:4322'+base+'papers/')
  page.locator('.item-card > a').first.click()
  assert page.locator('h1').count()==1
  page.screenshot(path=f'artifacts/{lang}-paper.png')
  page.close()
 b.close()

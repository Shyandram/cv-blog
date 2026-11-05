"""Interact with the persistent reading console through its public buttons."""
def set_reader_group(page, kind, opened):
    selector = '.chapter-section' if kind == 'sections' else '.code-fold'
    total = page.locator(selector).count()
    count = page.locator(selector + '[open]').count()
    button = page.locator('.reader-console-fold-' + kind)
    if opened and count == total or not opened and count == 0:
        return
    if count:
        button.click()
    if opened:
        button.click()
    page.wait_for_function('(a)=>document.querySelectorAll(a.selector + "[open]").length===a.count',
                           arg={'selector': selector, 'count': total if opened else 0})

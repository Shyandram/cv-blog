"""Check the production CSP and browser behavior after `npm run build`.

Run with `.venv/bin/python scripts/check-security.py` (Playwright is required).
The local server emulates public/_headers; verify the real headers after deploy.
"""
import base64
import fnmatch
import hashlib
import json
import re
import threading
from functools import partial
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.policy = ""
        self.redirect = False
        self.inline_scripts = []
        self.script = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta":
            kind = attrs.get("http-equiv", "").lower()
            if kind == "content-security-policy":
                self.policy = attrs.get("content", "")
            if kind == "refresh":
                self.redirect = True
        if tag == "script" and "src" not in attrs:
            if attrs.get("type", "") in ("", "module", "text/javascript"):
                self.script = ""

    def handle_data(self, data):
        if self.script is not None:
            self.script += data

    def handle_endtag(self, tag):
        if tag == "script" and self.script is not None:
            self.inline_scripts.append(self.script)
            self.script = None


def check_build():
    assert (DIST / "en/index.html").is_file(), "Run npm run build first"
    pages = 0
    for path in DIST.rglob("*.html"):
        html = path.read_text()
        doc = Document()
        doc.feed(html)
        # Astro's legacy redirects contain only a meta refresh and fallback link.
        if doc.redirect and not doc.inline_scripts:
            continue
        assert doc.policy, f"Missing CSP: {path}"
        directives = {d.split()[0]: d.split()[1:] for d in doc.policy.split(";") if d.strip()}
        scripts = directives.get("script-src", [])
        assert "'unsafe-inline'" not in scripts and "'unsafe-eval'" not in scripts, path
        assert directives.get("default-src") == ["'self'"], path
        for script in doc.inline_scripts:
            digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
            assert f"'sha256-{digest}'" in scripts, f"Unapproved inline script: {path}"
        assert "fonts.googleapis.com" not in html and "fonts.gstatic.com" not in html, path
        pages += 1
    return pages


def header_rules():
    rules = []
    for line in (DIST / "_headers").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            rules.append((line, {}))
        else:
            key, value = line.strip().split(":", 1)
            rules[-1][1][key] = value.strip()
    return rules


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        for pattern, headers in header_rules():
            if fnmatch.fnmatch(urlsplit(self.path).path, pattern):
                for key, value in headers.items():
                    self.send_header(key, value)
        super().end_headers()

    def log_message(self, *args):
        pass


def check_browser(origin):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        violations, errors, external = [], [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda msg: violations.append(re.sub(r"data:[^'\s]+", "data:[omitted]", msg.text)) if msg.type == "error" else None)
        page.on("request", lambda req: external.append(req.url) if urlsplit(req.url).scheme in ("http", "https") and not req.url.startswith(origin + "/") else None)
        for lang in ["en", "zh-tw"]:
            for width in [1440, 375]:
                page.set_viewport_size({"width": width, "height": 1000})
                page.goto(f"{origin}/{lang}/")
                page.evaluate("document.fonts.ready")
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (lang, width)
                page.screenshot(path=f"/tmp/cv-blog-security-{lang}-{width}.png")
                page.locator("#theme-toggle").click()
                theme = page.evaluate("document.documentElement.dataset.theme")
                page.reload()
                assert page.evaluate("document.documentElement.dataset.theme") == theme
            page.goto(f"{origin}/{lang}/learn/ml/ml-03-convolutional-networks/")
            page.evaluate("document.fonts.ready")
            assert page.locator(".katex").count() > 0
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.goto(f"{origin}/{lang}/search/")
            page.locator(".pagefind-ui__search-input").fill("convolution" if lang == "en" else "卷積")
            page.locator(".pagefind-ui__result-link").first.wait_for()
            assert page.locator(".pagefind-ui__result-link").count() > 0
            page.goto(f"{origin}/{lang}/playground/")
            page.locator("details.experiment-panel").evaluate_all("items => items.forEach(item => item.open = true)")
            fields = page.locator("experiment-playground input[type=text], experiment-playground textarea")
            assert fields.count() >= 4
            for field in fields.all():
                # Cards load their computation module near the viewport.
                field.scroll_into_view_if_needed()
                expect(field).to_be_enabled()
                field.fill('<svg onload="window.__injected=1"></svg>')
            assert page.evaluate("window.__injected === undefined")
            assert page.locator(".experiment-plot script, .experiment-plot [onload]").count() == 0
            reset = page.locator("experiment-playground [data-reset]").first
            reset.scroll_into_view_if_needed()
            expect(reset).to_be_enabled()
            reset.click()
        page.goto(f"{origin}/en/")
        page.locator("a.lang").click()
        page.wait_for_url("**/zh-tw/")
        page.goto(origin + "/")
        page.wait_for_url("**/zh-tw/")
        assert not errors, errors
        assert not violations, violations
        assert not external, external
        # The actual browser must reject both an injected script and string eval.
        blocked = page.evaluate("""async () => {
          const violations = [];
          document.addEventListener('securitypolicyviolation', e => violations.push(e.effectiveDirective));
          const script = document.createElement('script');
          script.textContent = 'window.__injected = 1';
          document.body.append(script);
          await new Promise(resolve => setTimeout(resolve, 100));
          return { untouched: window.__injected === undefined, violations };
        }""")
        assert blocked["untouched"], blocked
        assert any(d.startswith("script-src") for d in blocked["violations"]), blocked
        # DevTools normally bypasses eval restrictions. Explicitly disable that
        # privilege so this check uses the same CSP restrictions as page code.
        cdp = context.new_cdp_session(page)
        evaluated = cdp.send("Runtime.evaluate", {
            "expression": "(() => { try { eval('window.__injected = 2'); return false; } catch (error) { return error instanceof EvalError; } })()",
            "allowUnsafeEvalBlockedByCSP": False,
            "returnByValue": True,
        })
        assert evaluated.get("result", {}).get("value") is True, evaluated
        assert page.evaluate("window.__injected === undefined")
        cdp.detach()
        browser.close()


if __name__ == "__main__":
    pages = check_build()
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(DIST)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        check_browser(f"http://127.0.0.1:{server.server_port}")
    finally:
        server.shutdown()
        server.server_close()
    print(json.dumps({"pages_with_checked_csp": pages, "browser_security_checks": "passed"}))

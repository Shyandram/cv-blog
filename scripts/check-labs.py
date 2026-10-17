"""Browser smoke test for the image labs in the Playground (needs a built site in dist/).

For every lab it checks, in English and Traditional Chinese:
  * the lab loads, draws every canvas (not blank) and fills the metrics, with no console errors;
  * the default state already shows the idea (a metric threshold per lab, see EXPECT);
  * the predict step is reachable by keyboard and reports a verdict;
  * at 375 px wide nothing scrolls horizontally.

Run:  .venv/bin/python scripts/check-labs.py            (serves dist/ on a free port)
      LAB_BASE=http://localhost:4321 .venv/bin/python scripts/check-labs.py   (use a running server)
Chromium comes from PLAYWRIGHT_BROWSERS_PATH (do not run `playwright install`).
"""
import glob
import http.server
import os
import re
import socketserver
import sys
import threading
from functools import partial
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
# Labs are discovered from the lazy loaders; a lab without an entry in EXPECT only gets the generic checks.
LABS = re.findall(r"^\s*'([a-z0-9-]+)':\s*\(\)\s*=>", (ROOT / "src/lib/playground/labs/loaders.ts").read_text(), re.M)


def pct(s):
    return float(re.sub(r"[^0-9.\-]", "", s))


# Each check receives a function metrics() -> {label: text} and set(key, value) to move a control; it raises on failure.
def check_frequency_filter(metrics, set_, zh):
    m = metrics()
    out = [v for k, v in m.items() if "%" in v and ("out of range" in k.lower() or "超出" in k)]
    assert out and pct(out[0]) > 0.5, f"ideal filter should ring by default: {m}"
    set_("filter", "gaussian")
    out = [v for k, v in metrics().items() if "%" in v and ("out of range" in k.lower() or "超出" in k)]
    assert pct(out[0]) < 0.05, f"gaussian must not ring: {metrics()}"


def check_segmentation(metrics, set_, zh):
    vals = [pct(v) for v in metrics().values()]
    assert vals[0] < 0.3 and vals[1] > 0.8, f"default ramp should break global but not local: {metrics()}"
    set_("ramp", "0")
    vals0 = [pct(v) for v in metrics().values()]
    assert vals0[0] > vals[0] + 0.2, f"global should do better under even light: {vals0} vs {vals}"


def check_lowlight(metrics, set_, zh):
    def snr():
        return [pct(v) for k, v in metrics().items() if "dB" in k or "訊雜" in k][0]
    set_("gain", "1"); a = snr()
    set_("gain", "4"); b = snr()
    assert abs(a - b) < 0.05, f"gain must not change SNR: {a} vs {b}"
    set_("photons", "100"); c = snr()
    assert c > a + 5, f"more photons must improve SNR: {a} -> {c}"


def check_generic(metrics, set_, zh):
    assert metrics(), "metrics should be filled"


EXPECT = {"frequency-filter": check_frequency_filter, "segmentation-threshold": check_segmentation, "lowlight-photons": check_lowlight}


def serve():
    base = os.environ.get("LAB_BASE")
    if base:
        return base, None
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args, **kwargs):
            pass

    handler = partial(Quiet, directory=str(ROOT / "dist"))
    srv = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}", srv


def main():
    base, srv = serve()
    exe = next(iter(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux*/chrome")), None)
    failures = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe, args=["--no-sandbox"])
        for lang in ("en", "zh-tw"):
            for lab_id in LABS:
                for width in (1280, 375):
                    page = browser.new_page(viewport={"width": width, "height": 900})
                    errors = []
                    page.on("console", lambda m, e=errors: e.append(m.text) if m.type == "error" else None)
                    page.on("pageerror", lambda e, errs=errors: errs.append(str(e)))
                    tag = f"{lang}/{lab_id}@{width}"
                    try:
                        page.goto(f"{base}/{lang}/playground/#{lab_id}", wait_until="load")
                        root = page.locator(f'image-lab[data-lab="{lab_id}"]')
                        root.scroll_into_view_if_needed()
                        page.wait_for_function("el => el.querySelector('[data-status]').hidden", arg=root.element_handle(), timeout=20000)
                        page.wait_for_timeout(300)
                        blank = page.evaluate("""el => [...el.querySelectorAll('canvas')].filter(c => {
                            const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
                            let lo = 255, hi = 0; for (let i = 0; i < d.length; i += 4) { lo = Math.min(lo, d[i]); hi = Math.max(hi, d[i]); }
                            return hi - lo < 8; }).length""", root.element_handle())
                        assert blank == 0, f"{blank} blank canvas(es)"
                        assert page.evaluate("document.documentElement.scrollWidth") <= width, "page scrolls horizontally"

                        def metrics():
                            return dict(zip(*[iter(root.locator("[data-metrics] dt, [data-metrics] dd").all_inner_texts())] * 2))

                        def set_(key, value):
                            root.locator(f'[data-key="{key}"]').evaluate(
                                "(el, v) => { el.value = v; el.dispatchEvent(new Event('input', {bubbles: true})); el.dispatchEvent(new Event('change', {bubbles: true})); }", value)
                            page.wait_for_timeout(250)

                        if width == 1280:
                            EXPECT.get(lab_id, check_generic)(metrics, set_, lang == "zh-tw")
                            # predict step by keyboard
                            first = root.locator('predict-reveal input[type=radio]').first
                            first.focus(); page.keyboard.press("Space")
                            root.locator("predict-reveal [data-check]").first.wait_for(state="visible")
                            assert not root.locator("predict-reveal [data-check]").first.is_disabled(), "check button should enable after choosing"
                            root.locator("predict-reveal [data-check]").first.focus(); page.keyboard.press("Enter")
                            verdict = root.locator("predict-reveal [data-verdict]").first.inner_text()
                            assert verdict.strip(), "predict verdict should appear"
                        assert not errors, f"console errors: {errors}"
                        print("ok  ", tag)
                    except Exception as exc:  # noqa: BLE001
                        failures.append(f"{tag}: {exc}")
                        print("FAIL", tag, exc)
                        page.screenshot(path=str(ROOT / f"dist/_fail_{lang}_{lab_id}_{width}.png"), full_page=False)
                    finally:
                        page.close()
        browser.close()
    if srv:
        srv.shutdown()
    if failures:
        print(f"\n{len(failures)} failure(s)")
        sys.exit(1)
    print("\nAll lab checks passed.")


if __name__ == "__main__":
    main()

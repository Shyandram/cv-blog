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


def first_float(s):
    return float(re.search(r"-?\d+(?:\.\d+)?", s).group(0))


def check_transform_coding(metrics, set_, zh):
    m = list(metrics().values())  # nonzero %, PSNR, error smooth/busy halves, zoom box
    assert 5 < pct(m[0]) < 9 and 29 < pct(m[1]) < 32, f"default (DCT, step 0.2): about 7% nonzero, 30.5 dB: {m}"
    s, b = [float(x) for x in m[2].split("/")]
    assert b > 2 * s, f"the busy half should have over twice the error of the smooth half: {m}"
    set_("step", "0.6")
    m2 = list(metrics().values())
    assert pct(m2[0]) < 4 and pct(m2[1]) < 27, m2


def check_inverse_regularization(metrics, set_, zh):
    def restored():
        return [pct(v) for k, v in metrics().items() if "restored" in k.lower() or "還原後" in k][0]
    assert restored() < 10, f"the plain inverse filter should fail by default: {metrics()}"
    set_("logk", "-2.75")
    assert restored() > 25, f"a small K should win: {metrics()}"


def check_morphology_shapes(metrics, set_, zh):
    v = list(metrics().values())
    assert v[1].replace(" ", "") == "74→24" and v[2].replace(" ", "") == "50/50", f"opening removes all clutter by default: {v}"
    set_("shape", "hline")
    assert list(metrics().values())[2].replace(" ", "") == "45/50", metrics()


def check_array_dtype(metrics, set_, zh):
    v = list(metrics().values())  # wrapped %, clipped %, mean brightness, PSNR
    assert pct(v[0]) > 20 and pct(v[1]) == 0 and pct(v[2]) < 125, f"default should wrap the bright pixels: {metrics()}"
    set_("mode", "saturate")
    w = list(metrics().values())
    assert pct(w[0]) == 0 and pct(w[1]) > 20 and pct(w[2]) > 170, f"saturating arithmetic should clip, not wrap: {metrics()}"


def check_pixel_sampling(metrics, set_, zh):
    v = [pct(x) for x in list(metrics().values())[:3]]  # PSNR photo, PSNR zone plate, false-pattern contrast
    assert v[2] > 0.3 and v[1] < 8, f"default stride 4 should alias the zone plate: {metrics()}"
    set_("prefilter", "gaussian")
    w = [pct(x) for x in list(metrics().values())[:3]]
    assert w[2] < 0.1 and w[0] > v[0] + 1 and w[1] > v[1] + 1, f"the Gaussian prefilter should remove the false rings and raise PSNR: {metrics()}"


def check_color_clipping(metrics, set_, zh):
    v = [pct(x) for x in list(metrics().values())[:3]]  # overflow %, mean hue change, colour lost %
    assert v[0] > 60 and v[1] > 10 and v[2] > 5, f"default per-channel clipping should shift hue: {metrics()}"
    set_("method", "scale")
    w = [pct(x) for x in list(metrics().values())[:3]]
    assert w[1] < 0.5 and w[2] == 0, f"rescaling keeps hue: {metrics()}"


def check_corner_tensor(metrics, set_, zh):
    def r():
        return pct(next(v for k, v in metrics().items() if "Harris" in k))
    assert r() > 0.95, metrics()  # the default probe is the strongest corner
    set_("x", "30"); set_("y", "80")
    assert r() < -0.1, metrics()  # an edge has negative R
    set_("x", "200"); set_("y", "20")
    assert abs(r()) < 0.01, metrics()  # flat sky: R near 0


def check_receptive_field(metrics, set_, zh):
    def rf():
        return int(re.search(r"\d+", next(iter(metrics().values()))).group())
    assert rf() == 29, metrics()
    set_("stack", "plain3")
    assert rf() == 7, metrics()
    set_("stack", "stride-first")
    assert rf() == 11, metrics()


def check_neighbor(metrics, set_, zh):
    v = list(metrics().values())  # accuracy, accuracy in cm, nearest neighbours changed, tie-dependent
    assert first_float(v[1]) - first_float(v[0]) > 0.05 and int(v[2].split("/")[0]) > 20, metrics()
    set_("scale", "cm")
    v = list(metrics().values())
    assert int(v[2].split("/")[0]) == 0 and abs(first_float(v[0]) - first_float(v[1])) < 1e-9, metrics()


def check_diffusion(metrics, set_, zh):
    v = list(metrics().values())  # alpha, PSNR of the chosen target, PSNR eps/x0/v, error multiplier
    p = [float(x) for x in re.findall(r"\d+\.\d", v[2])]
    assert p[1] - p[0] > 15 and abs(first_float(v[3]) - 10) < 0.5, metrics()
    set_("err", "0.1")
    assert first_float(list(metrics().values())[1]) < p[0] - 2, metrics()


def check_bagging(metrics, set_, zh):
    v = [pct(x) for x in metrics().values()]  # bagged, single tree, average member, disagreement (all %)
    assert v[1] - v[0] > 1.5 and v[2] > v[1] and v[3] > 10, metrics()
    set_("depth", "2")
    w = [pct(x) for x in metrics().values()]
    assert w[3] < 6 and abs(w[1] - w[0]) < 0.5, metrics()  # stable shallow trees: no gain


def check_linear_transform(metrics, set_, zh):
    v = list(metrics().values())  # det, s1, s2, kappa, rank, orientation
    assert abs(first_float(v[0]) - 1) < 0.01 and abs(first_float(v[1]) - 1.62) < 0.01 and abs(first_float(v[2]) - 0.62) < 0.01 and abs(first_float(v[3]) - 2.62) < 0.01, v  # default shear: area kept, circle becomes an ellipse
    set_("preset", "stretch")
    set_("k", "-1")
    assert first_float(list(metrics().values())[0]) < -0.99, metrics()  # negative det: mirrored
    set_("k", "0")
    w = list(metrics().values())
    assert "∞" in w[3] and w[4].startswith("1"), w  # rank 1


def check_contour_energy(metrics, set_, zh):
    def dist():
        return float(re.search(r"[\d.]+", list(metrics().values())[1]).group())
    assert dist() > 15, metrics()      # default: stuck on neighbouring coins, 17.7 px
    set_("radius", "28")
    assert dist() < 1, metrics()       # found the rim, 0.5 px


def check_mlp(metrics, set_, zh):
    v = list(metrics().values())  # moons train, moons test, moons loss, circles train, circles test, circles loss, rises
    assert first_float(v[0]) >= 95 and first_float(v[3]) >= 95, metrics()
    set_("m", "1")
    v = list(metrics().values())
    assert first_float(v[0]) <= 87.5 and first_float(v[3]) <= 66.5, metrics()


def check_tracking(metrics, set_, zh):
    m = list(metrics().values())
    assert first_float(m[0]) == 2 and first_float(m[2]) < 0.65, f"default overlap-only should swap two identities: {m}"
    set_("lambda", "0.5")
    m2 = list(metrics().values())
    assert first_float(m2[0]) == 0 and first_float(m2[2]) > 0.8, f"mixing colour in should remove the swap: {m2}"


def check_image_fusion(metrics, set_, zh):
    v = [pct(x) for x in metrics().values()]  # PSNR average, PSNR fused, map agreement on textured pixels (%)
    assert 29 < v[0] < 31 and v[1] > 38.5 and v[1] - v[0] > 8, f"default: averaging blurry (~30 dB), local-detail rule sharp (~39 dB): {metrics()}"
    set_("window", "0.5")
    w = [pct(x) for x in metrics().values()]
    assert w[1] < v[1] - 1.5 and w[2] < v[2] - 5, f"a tiny window should make the decision map noisy: {metrics()}"
    set_("smooth", "2")
    assert pct(list(metrics().values())[1]) > w[1] + 1.5, f"smoothing the map should repair most of it: {metrics()}"


def check_lut_trilinear(metrics, set_, zh):
    def errs():
        m = list(metrics().values())  # table size, nearest mean / max, trilinear mean / max
        return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", m[1])], [float(x) for x in re.findall(r"\d+(?:\.\d+)?", m[2])]
    n, t = errs()
    assert n[0] > 8 and n[1] > 20 and t[0] < 0.5 and t[1] <= 5, f"default M=9 gradient: nearest bands, trilinear near-exact: {metrics()}"
    set_("m", "33")
    n33, t33 = errs()
    assert 1.5 < n33[0] < 3.5 and t33[0] < 0.1 and n33[0] > t[0] * 4, f"M=33: nearest still off by a couple of levels, more than trilinear at M=9: {metrics()}"
    set_("m", "2")
    n2, t2 = errs()
    assert t2[0] > 5 and n2[0] > t2[0], f"two nodes cannot follow the curve: {metrics()}"


def check_generic(metrics, set_, zh):
    assert metrics(), "metrics should be filled"


EXPECT = {
    "frequency-filter": check_frequency_filter, "segmentation-threshold": check_segmentation, "lowlight-photons": check_lowlight,
    "transform-coding": check_transform_coding, "inverse-regularization": check_inverse_regularization, "morphology-shapes": check_morphology_shapes,
    "array-dtype": check_array_dtype, "pixel-sampling": check_pixel_sampling, "color-clipping": check_color_clipping,
    "corner-tensor": check_corner_tensor, "receptive-field": check_receptive_field, "neighbor-geometry": check_neighbor, "diffusion-targets": check_diffusion,
    "bagging-trees": check_bagging, "tracking-association": check_tracking, "image-fusion": check_image_fusion, "lut-trilinear": check_lut_trilinear, "mlp-training": check_mlp, "contour-energy": check_contour_energy, "linear-transform": check_linear_transform,
}


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

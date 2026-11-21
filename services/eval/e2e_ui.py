"""End-to-end UI test: drives the real StyleMate page in headless Edge/Chrome over the
DevTools protocol, like a user would, and reports every step plus all JS errors.

  services/.venv/Scripts/python services/eval/e2e_ui.py [photo.jpg] [--marker 15] [--height 0]

Needs mini.py on :5000 (AI services optional). Exit code 1 if any step fails.
"""
import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[2]
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]


class Page:
    def __init__(self, ws):
        self.ws, self.n, self.events, self.errors, self.failed = ws, 0, [], [], []

    async def send(self, method, **params):
        self.n += 1
        my = self.n
        await self.ws.send(json.dumps({"id": my, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == my:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
            self._event(msg)

    def _event(self, msg):
        m, p = msg.get("method"), msg.get("params", {})
        if m == "Runtime.exceptionThrown":
            d = p["exceptionDetails"]
            self.errors.append(f"JS exception: {d.get('exception', {}).get('description', d.get('text'))[:300]}")
        elif m == "Runtime.consoleAPICalled" and p.get("type") == "error":
            self.errors.append("console.error: " + " ".join(str(a.get("value", a.get("description", "")))[:200] for a in p["args"]))
        elif m == "Network.responseReceived" and p["response"]["status"] >= 400:
            self.failed.append(f"{p['response']['status']} {p['response']['url']}")
        elif m == "Network.loadingFailed" and not p.get("canceled"):
            self.failed.append(f"FAILED {p.get('errorText')} ({p.get('type')})")

    async def pump(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            try:
                self._event(json.loads(await asyncio.wait_for(self.ws.recv(), timeout=max(0.05, end - time.time()))))
            except asyncio.TimeoutError:
                break

    async def js(self, expr, await_promise=True):
        r = await self.send("Runtime.evaluate", expression=expr, awaitPromise=await_promise, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"].get("exception", {}).get("description", "eval error")[:300])
        return r.get("result", {}).get("value")

    async def wait_for(self, expr, timeout=60, every=0.5):
        end = time.time() + timeout
        while time.time() < end:
            v = await self.js(expr)
            if v:
                return v
            await self.pump(every)
        return None


async def run(args):
    exe = next((b for b in BROWSERS if os.path.exists(b)), None)
    if not exe:
        print("No Edge/Chrome found"); return 1
    profile_dir = tempfile.mkdtemp(prefix="sm_e2e_")
    port = 9333
    proc = subprocess.Popen([exe, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={profile_dir}",
                             "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--window-size=1280,1600",
                             "--no-first-run", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results = []

    def step(name, ok, detail=""):
        results.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))

    try:
        for _ in range(60):
            try:
                targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                break
            except Exception:
                time.sleep(0.5)
        target = next(t for t in targets if t["type"] == "page")
        async with websockets.connect(target["webSocketDebuggerUrl"], max_size=50_000_000) as ws:
            pg = Page(ws)
            for d in ("Runtime", "Page", "Network", "DOM"):
                await pg.send(f"{d}.enable")
            t0 = time.time()
            await pg.send("Page.navigate", url=f"{args.base}/?e2e={int(t0)}")
            ok = await pg.wait_for("document.readyState === 'complete' && !!window.AvatarStudio", 30)
            step("page loads", ok, f"{time.time() - t0:.1f}s")
            ok = await pg.wait_for("window.AvatarStudio.ready", 90)
            step("3D avatar loads", ok)
            ok = await pg.wait_for("document.querySelectorAll('#rec-groups .product').length > 0", 30)
            n_cards = await pg.js("document.querySelectorAll('#rec-groups .product').length")
            step("catalog recommendations shown before any photo", ok, f"{n_cards} cards")
            svc = await pg.js("[...document.querySelectorAll('#svc-strip .svc')].map(e => e.textContent.trim() + ':' + (e.classList.contains('on') ? 'on' : 'off')).join(', ')")
            step("service status strip", bool(svc), svc)

            # ---- upload + analyze
            doc = await pg.send("DOM.getDocument")
            node = await pg.send("DOM.querySelector", nodeId=doc["root"]["nodeId"], selector="#file-input")
            await pg.send("DOM.setFileInputFiles", nodeId=node["nodeId"], files=[str(Path(args.photo).resolve())])
            await pg.pump(0.5)
            await pg.js(f"document.getElementById('marker-input').value = '{args.marker}'; "
                        f"document.getElementById('height-input').value = '{args.height or ''}'; true")
            t0 = time.time()
            await pg.js("document.getElementById('analyze-btn').click(); true")
            ok = await pg.wait_for("document.getElementById('results').classList.contains('visible') && "
                                   "document.getElementById('res-size').textContent.trim().length > 1", 120)
            prof_ui = await pg.js("""({size: document.getElementById('res-size').textContent.trim(),
                age: document.getElementById('res-age').textContent.trim(), height: document.getElementById('res-height').textContent.trim(),
                face: document.getElementById('res-face-shape').textContent.trim(), conf: document.getElementById('conf-text').textContent,
                msg: document.getElementById('upload-msg').textContent, notes: document.getElementById('res-notes').innerText})""")
            step("analyze photo", ok, f"{time.time() - t0:.1f}s · {json.dumps(prof_ui, ensure_ascii=False)[:400]}")
            ok = await pg.wait_for("[...document.querySelectorAll('#rec-groups .product .size')].some(e => e.textContent.includes('Your size'))", 20)
            step("personalised picks show size advice", ok,
                 await pg.js("[...document.querySelectorAll('#rec-groups .product .size')].filter(e => e.textContent.includes('Your size')).slice(0,3).map(e=>e.textContent.trim()).join(' | ')"))

            # ---- filters / search / explore
            await pg.js("document.getElementById('budget-input').value='800'; document.getElementById('budget-input').dispatchEvent(new Event('change')); true")
            await pg.pump(3)
            prices = await pg.js(r"[...document.querySelectorAll('#rec-groups .product .price')].map(e => (e.textContent.match(/₹\s*([\d,]+)/) || [])[1]).filter(Boolean).map(t => parseInt(t.replace(/,/g, '')))")
            note = await pg.js("document.getElementById('filter-note').textContent")
            step("budget filter (<= 800)", prices and max(prices) <= 800, f"max shown {max(prices) if prices else '-'}, note: {note}")
            await pg.js("document.getElementById('budget-input').value=''; document.getElementById('budget-input').dispatchEvent(new Event('change')); true")
            await pg.pump(3)
            await pg.js("document.getElementById('search-input').value='black shoes'; document.getElementById('search-btn').click(); true")
            ok = await pg.wait_for("document.querySelectorAll('#explore-body .product').length > 0", 20)
            step("search 'black shoes'", ok, await pg.js("document.getElementById('explore-title').textContent"))
            await pg.js("document.querySelector('#rec-groups .product [data-act=similar]').click(); true")
            ok = await pg.wait_for("document.getElementById('explore-title').textContent.startsWith('More like') && document.querySelectorAll('#explore-body .product').length > 0", 20)
            step("similar items", ok)
            await pg.js("document.querySelector('#rec-groups .product [data-act=complete]').click(); true")
            ok = await pg.wait_for("document.getElementById('explore-title').textContent.startsWith('Goes with') && document.querySelectorAll('#explore-body .product').length > 0", 20)
            step("complete the look", ok)

            # ---- 3D try-on: wear one item from every category shown
            cats = await pg.js("[...document.querySelectorAll('#rec-groups .cat-title')].map(e => e.textContent.trim())")
            worn = []
            for cat in cats:
                await pg.js(f"""(() => {{ const t = [...document.querySelectorAll('#rec-groups .cat-title')].find(e => e.textContent.trim() === '{cat}');
                    t.nextElementSibling.querySelector('[data-act=wear]').click(); return true; }})()""")
                await pg.pump(1.5)
                worn.append(await pg.js("AvatarStudio.wornItems().map(w => w.slot).join(',')"))
            slots = await pg.js("AvatarStudio.wornItems().map(w => w.slot)")
            step("wear one item per category in 3D", len(slots) >= 3, f"worn now: {slots}")
            pen = await pg.js("AvatarStudio.settle(1); AvatarStudio.clothPenetration()")
            step("no cloth inside the body", pen == 0, f"{pen} points")
            ok = await pg.wait_for("document.querySelectorAll('#worn-list .worn').length > 0", 5)
            step("'Wearing' panel lists items", ok, await pg.js("document.getElementById('worn-list').innerText.replace(/\\n+/g,' / ').slice(0,200)"))
            fit = await pg.wait_for("document.getElementById('fit-panel').style.display !== 'none' && document.getElementById('fit-summary').textContent.length > 3 && document.getElementById('fit-summary').textContent", 15)
            step("fit panel for the last garment", bool(fit), (fit or "")[:160])
            if fit:
                n_opts = await pg.js("document.getElementById('fit-size').options.length")
                await pg.js("const s=document.getElementById('fit-size'); s.selectedIndex = 0; s.dispatchEvent(new Event('change')); true")
                await pg.pump(2)
                step("change size in fit panel", n_opts > 1, (await pg.js("document.getElementById('fit-summary').textContent"))[:120])
            await pg.js("document.querySelector('#worn-list [data-wact=remove]').click(); true")
            await pg.pump(1)
            step("take an item off", len(await pg.js("AvatarStudio.wornItems()")) == len(slots) - 1)

            # ---- outfit save / list / delete
            await pg.js("document.querySelector('#rec-groups .product [data-act=select]').click(); true")
            await pg.pump(0.5)
            before = await pg.js("document.querySelectorAll('#saved-outfits-list .saved-item').length")
            await pg.js("document.getElementById('save-outfit-btn').click(); true")
            ok = await pg.wait_for(f"document.querySelectorAll('#saved-outfits-list .saved-item').length > {before}", 15)
            step("save outfit", ok)
            if ok:
                await pg.js("document.querySelector('#saved-outfits-list .saved-item:last-child [data-sact=del]').click(); true")
                ok = await pg.wait_for(f"document.querySelectorAll('#saved-outfits-list .saved-item').length === {before}", 15)
                step("delete saved outfit", ok)
            # ---- body / scene controls
            await pg.js("document.getElementById('phys-slim').click(); document.getElementById('skin-btn').click(); "
                        "const w=document.getElementById('wind-input'); w.value='1'; w.dispatchEvent(new Event('input')); "
                        "document.getElementById('cam-face').click(); document.getElementById('spin-btn').click(); "
                        "document.getElementById('reset-btn').click(); true")
            await pg.pump(2)
            step("body, skin, wind, camera, reset controls", (await pg.js("AvatarStudio.wornItems().length")) == 0)
            await pg.pump(1)
            step("no JavaScript errors", not pg.errors, "; ".join(pg.errors[:5]))
            step("no failed requests", not pg.failed, "; ".join(sorted(set(pg.failed))[:6]))
    finally:
        proc.terminate()
        try:
            proc.wait(5)
        except Exception:
            proc.kill()
        shutil.rmtree(profile_dir, ignore_errors=True)
    fails = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(fails)}/{len(results)} steps passed")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("photo", nargs="?", default=str(ROOT / "services/eval_results/height/child_marker.jpg"))
    ap.add_argument("--marker", default="15")
    ap.add_argument("--height", default="")
    ap.add_argument("--base", default="http://127.0.0.1:5000")
    sys.exit(asyncio.run(run(ap.parse_args())))

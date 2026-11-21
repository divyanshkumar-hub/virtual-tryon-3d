"""Load pages in headless Edge/Chrome and report JS errors / failed requests per page.
  services/.venv/Scripts/python services/eval/smoke_pages.py URL [URL ...]"""
import asyncio, json, os, shutil, subprocess, sys, tempfile, time, urllib.request
import websockets
sys.path.insert(0, os.path.dirname(__file__))
from e2e_ui import BROWSERS, Page  # noqa: E402


async def main(urls):
    exe = next(b for b in BROWSERS if os.path.exists(b))
    prof = tempfile.mkdtemp(prefix="sm_smoke_")
    proc = subprocess.Popen([exe, "--headless=new", "--remote-debugging-port=9334", f"--user-data-dir={prof}",
                             "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-first-run", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bad = 0
    try:
        for _ in range(60):
            try:
                t = next(x for x in json.load(urllib.request.urlopen("http://127.0.0.1:9334/json")) if x["type"] == "page"); break
            except Exception:
                time.sleep(0.5)
        async with websockets.connect(t["webSocketDebuggerUrl"], max_size=50_000_000) as ws:
            pg = Page(ws)
            for d in ("Runtime", "Page", "Network"):
                await pg.send(f"{d}.enable")
            for url in urls:
                pg.errors, pg.failed = [], []
                await pg.send("Page.navigate", url=url)
                await pg.wait_for("document.readyState === 'complete'", 30)
                await pg.pump(8)
                title = await pg.js("document.title")
                ok = not pg.errors and not pg.failed
                bad += not ok
                print(f"[{'PASS' if ok else 'FAIL'}] {url}  ({title})")
                for e in pg.errors[:5] + sorted(set(pg.failed))[:5]:
                    print("      ", e)
    finally:
        proc.terminate(); shutil.rmtree(prof, ignore_errors=True)
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))

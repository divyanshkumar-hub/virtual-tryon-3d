"""Download small labeled evaluation samples from public Hugging Face datasets.

  python services/eval/fetch_data.py            (stdlib only)

Writes services/eval_data/{utkface,fashion}/ + labels.json
"""
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "eval_data"
API = "https://datasets-server.huggingface.co/rows?dataset={ds}&config=default&split=train&offset={off}&length={n}"
random.seed(7)


def rows(ds, off, n):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(API.format(ds=ds, off=off, n=n), timeout=60) as r:
                return [x["row"] for x in json.load(r)["rows"]]
        except Exception as exc:  # noqa: BLE001
            time.sleep(2 + attempt * 3)
            err = exc
    raise err


def save(url, path):
    for attempt in range(6):
        if path.exists():
            return
        try:
            urllib.request.urlretrieve(url, path)
        except Exception:  # noqa: BLE001  (429 rate limit -> back off)
            time.sleep(3 * (attempt + 1))


def fetch(ds, total, chunks, per, folder, keep):
    d = OUT / folder
    d.mkdir(parents=True, exist_ok=True)
    offsets = sorted(random.sample(range(0, total - per), chunks))
    items, jobs = [], []
    for off in offsets:
        time.sleep(0.5)
        for r in rows(ds, off, per):
            name = f"{folder}_{off}_{len(items)}.jpg"
            jobs.append((r["image"]["src"], d / name))
            items.append({"file": name, **{k: r[k] for k in keep}})
    with ThreadPoolExecutor(3) as ex:
        list(ex.map(lambda j: save(*j), jobs))
    items = [it for it in items if (d / it["file"]).exists()]
    (d / "labels.json").write_text(json.dumps(items, indent=1))
    print(folder, len(items))


if __name__ == "__main__":
    fetch("nu-delta/utkface", 23705, 40, 5, "utkface", ["age", "gender", "ethnicity"])
    fetch("ashraq/fashion-product-images-small", 44072, 60, 10, "fashion",
          ["articleType", "baseColour", "usage", "gender", "subCategory", "productDisplayName"])

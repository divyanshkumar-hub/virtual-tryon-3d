"""Sample labeled eval sets from downloaded parquet shards (needs pyarrow; run with the services venv).

  services/.venv/Scripts/python services/eval/prepare_data.py
"""
import io
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

D = Path(__file__).resolve().parent.parent / "eval_data"
random.seed(7)


def dump(parquet, folder, n, keep, stratify=None):
    t = pq.read_table(D / "raw" / parquet).to_pylist()
    if stratify:  # even spread over a label (e.g. age decade) instead of the dataset's skew
        groups = {}
        for r in t:
            groups.setdefault(stratify(r), []).append(r)
        per = max(1, n // len(groups))
        t = [r for g in sorted(groups) for r in random.sample(groups[g], min(per, len(groups[g])))]
    else:
        t = random.sample(t, min(n, len(t)))
    out = D / folder
    out.mkdir(parents=True, exist_ok=True)
    labels = []
    for i, r in enumerate(t):
        name = f"{folder}_{i:04d}.jpg"
        Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(out / name, quality=95)
        labels.append({"file": name, **{k: r[k] for k in keep}})
    (out / "labels.json").write_text(json.dumps(labels, indent=1))
    print(folder, len(labels))


if __name__ == "__main__":
    if (D / "raw" / "utk1.parquet").exists():
        dump("utk1.parquet", "utkface", 240, ["age", "gender", "ethnicity"],
             stratify=lambda r: min(r["age"] // 10, 8))
    if (D / "raw" / "fashion0.parquet").exists():
        dump("fashion0.parquet", "fashion", 600,
             ["articleType", "baseColour", "usage", "gender", "subCategory", "productDisplayName"])

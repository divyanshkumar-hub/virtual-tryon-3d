"""Hit the running services over HTTP and score them against labeled data.

  services/.venv/Scripts/python services/eval/run_evals.py fashion age height

Results -> services/eval_results/<name>.json (+ images for height).
"""
import collections
import json
import statistics as st
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "services" / "eval_data"
R = ROOT / "services" / "eval_results"
R.mkdir(exist_ok=True)


def post(port, path, files=None, data=None, js=None, timeout=600):
    r = requests.post(f"http://127.0.0.1:{port}{path}", files=files, data=data, json=js, timeout=timeout)
    r.raise_for_status()
    return r.json()


def img(path):
    return ("f.jpg", open(path, "rb"), "image/jpeg")


# ------------------------------------------------------------------ fashion
TYPE_MAP = {"Tshirts": "t-shirt", "Shirts": "shirt", "Sweaters": "sweater", "Sweatshirts": "hoodie",
            "Jackets": "jacket", "Blazers": "blazer", "Dresses": "dress", "Skirts": "skirt", "Jeans": "jeans",
            "Trousers": "trousers", "Shorts": "shorts", "Track Pants": "track pants", "Kurtas": "kurta",
            "Sarees": "saree", "Casual Shoes": "sneakers", "Sports Shoes": "sneakers", "Formal Shoes": "formal shoes",
            "Sandals": "sandals", "Flip Flops": "sandals", "Heels": "heels", "Handbags": "handbag",
            "Watches": "watch", "Sunglasses": "sunglasses", "Caps": "cap"}
COLOUR_MAP = {"Black": "black", "White": "white", "Grey": "grey", "Navy Blue": "navy blue", "Blue": "blue",
              "Red": "red", "Maroon": "maroon", "Pink": "pink", "Purple": "purple", "Green": "green",
              "Olive": "olive", "Yellow": "yellow", "Orange": "orange", "Brown": "brown", "Beige": "beige",
              "Cream": "cream", "Gold": "gold", "Silver": "silver", "Multi": "multicolour"}
USAGE_MAP = {"Casual": "casual", "Formal": "formal", "Sports": "sportswear", "Ethnic": "ethnic wear",
             "Party": "party wear"}
FABRIC_WORDS = ["denim", "linen", "silk", "satin", "wool", "leather", "suede", "chiffon", "velvet", "fleece",
                "corduroy", "lace", "cotton", "knit", "polyester", "nylon"]


def eval_fashion(port):
    labels = json.loads((D / "fashion" / "labels.json").read_text())
    acc = collections.defaultdict(lambda: [0, 0])  # metric -> [hits, n]
    confusions = collections.Counter()
    for it in labels:
        tags = post(port, "/tag", files={"file": img(D / "fashion" / it["file"])})
        for metric, truth, pred in (
            ("type", TYPE_MAP.get(it["articleType"]), tags["type"]),
            ("colour", COLOUR_MAP.get(it["baseColour"]), tags["colour"]),
            ("style", USAGE_MAP.get(it["usage"]), tags["style"]),
        ):
            if truth is None:
                continue
            top = [p["label"] for p in pred]
            acc[f"{metric}_top1"][0] += top[0] == truth
            acc[f"{metric}_top3"][0] += truth in top
            acc[f"{metric}_top1"][1] += 1
            acc[f"{metric}_top3"][1] += 1
            if metric == "type" and top[0] != truth:
                confusions[f"{truth} -> {top[0]}"] += 1
        name = (it["productDisplayName"] or "").lower()
        fab = next((w for w in FABRIC_WORDS if w in name), None)
        if fab:  # weak label: fabric word in the product title
            top = [p["label"] for p in tags["fabric"]]
            acc["fabric_top1 (weak, from title)"][0] += top[0] == fab
            acc["fabric_top3 (weak, from title)"][0] += fab in top
            acc["fabric_top1 (weak, from title)"][1] += 1
            acc["fabric_top3 (weak, from title)"][1] += 1
        post(port, "/catalog/add", files={"file": img(D / "fashion" / it["file"])},
             data={"item_id": it["file"], "name": it["productDisplayName"] or "", "category": it["subCategory"]})

    # retrieval: do the 5 nearest neighbours share the article type?
    by_id = {it["file"]: it for it in labels}
    p5 = []
    for it in labels[:200]:
        sims = requests.get(f"http://127.0.0.1:{port}/similar/{it['file']}?k=5", timeout=60).json()
        p5.append(sum(by_id[s["id"]]["articleType"] == it["articleType"] for s in sims) / 5)
    # text search: "<colour> <type>" queries for combos that exist >=3 times
    combos = collections.Counter((it["baseColour"], it["articleType"]) for it in labels)
    q_p5 = []
    for (c, t), n in combos.most_common():
        if n < 3 or t not in TYPE_MAP or c not in COLOUR_MAP:
            continue
        res = post(port, "/search", js={"q": f"{COLOUR_MAP[c]} {TYPE_MAP[t]}", "k": 5})
        q_p5.append(sum(by_id[r["id"]]["articleType"] == t and by_id[r["id"]]["baseColour"] == c for r in res) / 5)
    out = {k: {"acc": round(h / n, 3), "n": n} for k, (h, n) in sorted(acc.items())}
    out["similar_item_same_type_p@5"] = {"acc": round(st.mean(p5), 3), "n": len(p5)}
    out["text_search_colour+type_p@5"] = {"acc": round(st.mean(q_p5), 3), "n": len(q_p5)}
    out["top_type_confusions"] = confusions.most_common(10)
    out["health"] = requests.get(f"http://127.0.0.1:{port}/health").json()
    return out


# ------------------------------------------------------------------ age
def eval_age(port=5101):
    labels = json.loads((D / "utkface" / "labels.json").read_text())
    rows = []
    for it in labels:
        r = post(port, "/estimate", files={"file": img(D / "utkface" / it["file"])}, data={"mode": "face"})
        rows.append({"true": it["age"], "gender": it["gender"], "mivolo": r["mivolo"]["age"],
                     "mivolo_gender": r["mivolo"]["gender"], "vit": r["vit_baseline"]["age"],
                     "vit_bucket": r["vit_baseline"]["bucket"], "ms": r["latency_ms"]})

    def mae(key, sub):
        return round(st.mean(abs(x[key] - x["true"]) for x in sub), 2) if sub else None

    groups = collections.defaultdict(list)
    for x in rows:
        groups["0-12" if x["true"] <= 12 else "13-19" if x["true"] <= 19 else "20-39" if x["true"] <= 39
               else "40-59" if x["true"] <= 59 else "60+"].append(x)
    within5 = lambda k: round(sum(abs(x[k] - x["true"]) <= 5 for x in rows) / len(rows), 3)  # noqa: E731
    g_ok = [x for x in rows if x["gender"] in ("male", "female", "Male", "Female")]
    out = {
        "n": len(rows),
        "mae": {"mivolo_v2": mae("mivolo", rows), "vit_baseline": mae("vit", rows)},
        "within_5_years": {"mivolo_v2": within5("mivolo"), "vit_baseline": within5("vit")},
        "mae_by_age_group": {g: {"n": len(v), "mivolo_v2": mae("mivolo", v), "vit_baseline": mae("vit", v)}
                             for g, v in sorted(groups.items())},
        "mivolo_gender_acc": round(sum(x["mivolo_gender"].lower() == x["gender"].lower() for x in g_ok)
                                   / max(1, len(g_ok)), 3) if g_ok else None,
        "median_latency_ms": st.median(x["ms"] for x in rows),
        "worst": sorted(rows, key=lambda x: -abs(x["mivolo"] - x["true"]))[:8],
    }
    return out


# ------------------------------------------------------------------ height
def eval_height(port=5102):
    from height_synth import make_cases  # local helper

    out = {"cases": []}
    for case in make_cases(R / "height"):
        data = {"marker_cm": case.get("marker_cm", 15.0)}
        if case.get("user_height_cm"):
            data["user_height_cm"] = case["user_height_cm"]
        try:
            r = post(port, "/estimate", files={"file": img(case["path"])}, data=data)
        except requests.HTTPError as exc:
            r = {"error": exc.response.status_code, "detail": exc.response.json().get("detail")}
        out["cases"].append({**{k: v for k, v in case.items() if k != "path"}, "file": Path(case["path"]).name,
                             "result": r})
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    for job in sys.argv[1:]:
        if job == "fashion":
            res = {"fashion-clip": eval_fashion(5103), "openai-clip-baseline": eval_fashion(5113)}
        else:
            res = {"age": eval_age, "height": eval_height}[job]()
        (R / f"{job}.json").write_text(json.dumps(res, indent=1, default=str))
        print(json.dumps(res, indent=1, default=str)[:4000])

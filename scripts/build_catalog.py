"""Build the demo product catalog (static/catalog/) from the public
`ashraq/fashion-product-images-small` dataset (Kaggle "Fashion Product Images (Small)").

  services/.venv/Scripts/python scripts/build_catalog.py [path/to/fashion0.parquet]

Picks a balanced set per (audience, article type) so recommendations have
something sensible for men, women and kids. Images are the dataset's 60x80
thumbnails - fine for cards; the try-on model wants bigger garment photos.
"""
import io
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import hashlib

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fit import ADULT_SIZES, KIDS_SIZES, product_size_chart  # noqa: E402
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "services" / "eval_data" / "raw" / "fashion0.parquet"
OUT = ROOT / "static" / "catalog"

CATEGORY = {
    "Tshirts": "tops", "Shirts": "tops", "Tops": "tops", "Sweaters": "tops", "Sweatshirts": "tops", "Kurtas": "tops",
    "Jackets": "outerwear", "Blazers": "outerwear",
    "Jeans": "bottoms", "Trousers": "bottoms", "Shorts": "bottoms", "Track Pants": "bottoms", "Skirts": "bottoms",
    "Capris": "bottoms", "Leggings": "bottoms",
    "Dresses": "dresses",
    "Casual Shoes": "footwear", "Sports Shoes": "footwear", "Formal Shoes": "footwear", "Sandals": "footwear",
    "Heels": "footwear", "Flip Flops": "footwear",
    "Watches": "accessories", "Sunglasses": "accessories", "Handbags": "accessories", "Caps": "accessories",
    "Belts": "accessories", "Backpacks": "accessories",
}
AUDIENCE = {"Men": "men", "Women": "women", "Boys": "boys", "Girls": "girls", "Unisex": "unisex"}
PER_GROUP = {"men": 4, "women": 4, "boys": 4, "girls": 4, "unisex": 2}
# NOTE: per (audience, articleType).

COLOUR_HEX = {
    "Black": "#111111", "White": "#f5f5f5", "Blue": "#2563eb", "Brown": "#7c4a2d", "Grey": "#8a8f98",
    "Red": "#dc2626", "Green": "#16a34a", "Pink": "#ec4899", "Navy Blue": "#1e3a8a", "Purple": "#7e22ce",
    "Silver": "#c0c4cc", "Beige": "#d6c3a3", "Yellow": "#facc15", "Gold": "#d4a017", "Orange": "#f97316",
    "Maroon": "#7f1d1d", "Olive": "#6b7d2a", "Cream": "#f3ead3", "Multi": "#a855f7", "Steel": "#71797e",
    "Charcoal": "#36454f", "Off White": "#efeae0", "Peach": "#ffcba4", "Skin": "#e8b796", "Lavender": "#b9a7e6",
    "Khaki": "#b8a66a", "Magenta": "#c026d3", "Grey Melange": "#a3a6ab", "Tan": "#c49a6c", "Teal": "#0f766e",
    "Bronze": "#a3712f", "Copper": "#b87333", "Mustard": "#d4a72c", "Rust": "#b7410e",
    "Turquoise Blue": "#06b6d4", "Burgundy": "#7a1f3d", "Metallic": "#9ca3af", "Mauve": "#b784a7",
    "Coffee Brown": "#4b3621", "Rose": "#f43f5e",
}

# Demo retail metadata. The dataset has no prices / stock / size charts, so these are generated
# deterministically per SKU and flagged as demo data (price_is_demo) - not real retail prices.
PRICE_INR = {  # (min, max) per article type, rupees
    "Tshirts": (399, 1499), "Shirts": (799, 2499), "Tops": (499, 1799), "Sweaters": (999, 2999),
    "Sweatshirts": (899, 2499), "Kurtas": (699, 2999), "Jackets": (1499, 5999), "Blazers": (2999, 8999),
    "Jeans": (999, 3499), "Trousers": (899, 2999), "Shorts": (399, 1299), "Track Pants": (599, 1999),
    "Skirts": (499, 1999), "Capris": (399, 1299), "Leggings": (299, 999), "Dresses": (799, 3999),
    "Casual Shoes": (1299, 4999), "Sports Shoes": (1499, 6999), "Formal Shoes": (1999, 5999), "Sandals": (499, 2499),
    "Heels": (999, 3999), "Flip Flops": (199, 899), "Watches": (999, 9999), "Sunglasses": (599, 3999),
    "Handbags": (999, 4999), "Caps": (299, 999), "Belts": (399, 1499), "Backpacks": (799, 3499),
}
AUDIENCE_WORDS = {"men", "women", "boys", "girls", "unisex", "kids", "kid's", "infant", "infants", "teens", "kids-boys",
                  "kids-girls", "boy's", "girl's", "men's", "women's"}
FIT_WORDS = [("skinny", "skinny"), ("slim", "slim"), ("oversized", "oversized"), ("relaxed", "relaxed"),
             ("loose", "loose"), ("regular", "regular"), ("straight", "regular")]
PATTERN_WORDS = [("stripe", "striped"), ("check", "checked"), ("floral", "floral"), ("polka", "polka dot"),
                 ("graphic", "printed"), ("print", "printed"), ("self design", "textured"), ("solid", "solid"),
                 ("plain", "solid"), ("colourblock", "colour-blocked")]
SHOE_ADULT = ["UK 6", "UK 7", "UK 8", "UK 9", "UK 10", "UK 11"]
SHOE_KIDS = ["UK 10K", "UK 11K", "UK 12K", "UK 13K", "UK 1", "UK 2", "UK 3", "UK 4"]


def rng_for(sku):
    return random.Random(int(hashlib.md5(sku.encode()).hexdigest()[:8], 16))


def brand_of(name):
    words = name.split()
    for i, w in enumerate(words):
        if w.lower() in AUDIENCE_WORDS and i > 0:
            return " ".join(words[:i])
    return words[0] if words else None


def keyword(name, table, default=None):
    low = name.lower()
    return next((v for k, v in table if k in low), default)


def sizes_for(category, audience, rng):
    if category == "accessories":
        return ["One size"]
    kids = audience in ("boys", "girls")
    if category == "footwear":
        run = SHOE_KIDS if kids else SHOE_ADULT
    else:
        run = KIDS_SIZES if kids else ADULT_SIZES
    if kids and category != "footwear":                  # kidswear ranges start small and run 6-10 sizes
        start = rng.randint(0, 3)
        run = run[start:start + rng.randint(6, 10)]
    else:                                                # trim the ends sometimes
        run = run[rng.randint(0, 1): len(run) - rng.randint(0, 1)]
    if len(run) > 3 and rng.random() < 0.25:             # one size sold out now and then
        run = [s for s in run if s != rng.choice(run[1:-1])]
    return run


random.seed(11)
rows = pq.read_table(SRC).to_pylist()
groups = defaultdict(list)
for r in rows:
    aud = AUDIENCE.get(r["gender"])
    if aud and r["articleType"] in CATEGORY and r["baseColour"] in COLOUR_HEX and r["productDisplayName"]:
        groups[(aud, r["articleType"])].append(r)

(OUT / "images").mkdir(parents=True, exist_ok=True)
items = []
for (aud, art), rs in sorted(groups.items()):
    for r in random.sample(rs, min(PER_GROUP[aud], len(rs))):
        name = f"{r['id']}.jpg"
        Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(OUT / "images" / name, quality=92)
        sku = str(r["id"])
        rng = rng_for(sku)
        lo, hi = PRICE_INR.get(art, (499, 2999))
        if aud in ("boys", "girls"):                     # kidswear is cheaper
            lo, hi = int(lo * 0.6), int(lo * 0.6 + (hi - lo) * 0.45)
        cat = CATEGORY[art]
        fit_type = keyword(r["productDisplayName"], FIT_WORDS, "regular") if cat in ("tops", "outerwear", "bottoms", "dresses") else None
        sizes = sizes_for(cat, aud, rng)
        items.append({
            "sku": sku,
            "brand": brand_of(r["productDisplayName"].strip()),
            "price": int(round(rng.uniform(lo, hi) / 100.0)) * 100 - 1,
            "price_is_demo": True,
            "currency": "INR",
            "available_sizes": sizes,
            "fit_type": fit_type,
            "pattern": keyword(r["productDisplayName"], PATTERN_WORDS),
            "size_chart": product_size_chart(cat, aud, sizes, fit_type or "regular"),
            "name": r["productDisplayName"].strip(),
            "category": CATEGORY[art],
            "article_type": art,
            "audience": aud,
            "colour_name": r["baseColour"],
            "color": COLOUR_HEX[r["baseColour"]],
            "usage": r["usage"] or "Casual",
            "season": r["season"],
            "image_path": f"/static/catalog/images/{name}",
        })

(OUT / "catalog.json").write_text(json.dumps(items, indent=1))
print(len(items), "items ->", OUT)

"""Garment fit: compares body measurements with a size chart.

Garment ("finished") measurements are typical high-street size charts, in cm.
Ease = garment - body. Each region gets a verdict that the 3D avatar shows as
a tight / good / loose heat-map and as how much the garment is inflated or
shrunk on the body.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

ADULT_SIZES = ["XS", "S", "M", "L", "XL", "XXL"]
KIDS_SIZES = ["2-3Y", "3-4Y", "4-5Y", "5-6Y", "6-7Y", "7-8Y", "8-9Y", "9-10Y", "10-11Y", "11-12Y", "12-13Y", "13-14Y"]
_KIDS_HEIGHT = {s: h for s, h in zip(KIDS_SIZES, [98, 104, 110, 116, 122, 128, 134, 140, 146, 152, 158, 164])}

# finished garment measurements per size (cm)
CHART = {
    "men": {"chest": [94, 100, 106, 112, 120, 128], "waist": [74, 80, 86, 92, 100, 108], "hip": [92, 98, 104, 110, 118, 126]},
    "women": {"chest": [88, 94, 100, 106, 114, 122], "waist": [66, 72, 78, 84, 92, 100], "hip": [94, 100, 106, 112, 120, 128]},
}
# ease (garment - body) per category and region for a regular fit:
# (tight below, good from, good to, loose above)
EASE = {
    "tops": {"chest": (2, 6, 14, 20), "waist": (0, 4, 18, 26)},
    "dresses": {"chest": (2, 5, 12, 18), "waist": (0, 3, 12, 18), "hip": (1, 4, 12, 18)},
    "bottoms": {"waist": (-1, 0, 5, 9), "hip": (1, 4, 11, 16)},  # trousers sit on the waist; little ease
}
EASE["outerwear"] = {"chest": (6, 10, 20, 28), "waist": (4, 8, 24, 32)}  # worn over other layers
# a top's waist is cut from its chest, not from the trouser chart
TOP_WAIST_DROP = {"men": 6, "women": 12}
REGIONS = {"tops": ["chest", "waist"], "outerwear": ["chest", "waist"], "dresses": ["chest", "waist", "hip"],
           "bottoms": ["waist", "hip"]}
# how a cut changes the finished measurements (cm, applied to every region of the garment)
FIT_ADJUST = {"slim": -4, "skinny": -5, "regular": 0, "relaxed": 6, "loose": 8, "oversized": 12}
APPAREL = set(REGIONS)


def _kids_chart(size: str) -> Dict[str, float]:
    h = _KIDS_HEIGHT[size]
    # body proportions of children by stature (approx.), plus regular-fit ease
    return {"chest": 0.42 * h + 9 + 8, "waist": 0.36 * h + 12 + 2, "hip": 0.44 * h + 8 + 6}


def garment_measurements(size: str, audience: str, category: str = "tops") -> Dict[str, float]:
    female = audience in ("women", "girls")
    if size in _KIDS_HEIGHT:
        g = {k: round(v, 1) for k, v in _kids_chart(size).items()}
    else:
        chart = CHART["women" if female else "men"]
        i = ADULT_SIZES.index(size)
        g = {k: float(v[i]) for k, v in chart.items()}
    if category in ("tops", "outerwear"):
        g["waist"] = g["chest"] - TOP_WAIST_DROP["women" if female else "men"]
        if category == "outerwear":
            g = {k: v + 6 for k, v in g.items()}
    return g


def product_size_chart(category: str, audience: str, sizes: List[str], fit_type: str = "regular") -> Dict[str, Dict[str, float]]:
    """Finished garment measurements for each size of one product, adjusted for its cut."""
    if category not in APPAREL:
        return {}
    adj = FIT_ADJUST.get((fit_type or "regular").lower(), 0)
    chart = {}
    for sz in sizes:
        if sz in _KIDS_HEIGHT or sz in ADULT_SIZES:
            g = garment_measurements(sz, audience, category)
            chart[sz] = {r: round(g[r] + adj, 1) for r in REGIONS[category]}
    return chart


def verdict(category: str, region: str, ease: float) -> str:
    tight, good_lo, good_hi, loose = EASE[category][region]
    if ease < tight:
        return "tight"
    if ease < good_lo:
        return "snug"
    if ease <= good_hi:
        return "good"
    if ease <= loose:
        return "relaxed"
    return "loose"


_SCORE = {"good": 0, "snug": 1, "relaxed": 1, "tight": 3, "loose": 2}


def fit_report(measurements: Dict[str, Any], category: str, audience: str, size: Optional[str] = None,
               is_child: bool = False, chart: Optional[Dict[str, Dict[str, float]]] = None,
               available: Optional[List[str]] = None, fit_type: Optional[str] = None) -> Dict[str, Any]:
    """chart: the product's own size chart (else the generic one); available: sizes in stock;
    fit_type: the cut (slim/regular/relaxed/...). Verdicts compare the ease with what the cut
    intends, so a relaxed shirt in your size is "good", not "too big"."""
    cut = (fit_type or "regular").lower()
    intended = FIT_ADJUST.get(cut, 0)
    regions = REGIONS.get(category)
    if not regions:
        return {"supported": False, "reason": f"no fit data for {category}"}
    body = {"chest": measurements.get("chest_cm"), "waist": measurements.get("waist_cm"), "hip": measurements.get("hip_cm")}
    sizes = KIDS_SIZES if (is_child or audience in ("boys", "girls")) else ADULT_SIZES
    candidates = sizes
    if sizes is KIDS_SIZES and measurements.get("height_cm"):
        # kidswear is sized by stature; girths from a photo only fine-tune within +/-1 size
        h = float(measurements["height_cm"])
        # same rule as vision_analyzer.get_kids_size: smallest size whose nominal height covers the child
        i = next((j for j, sz in enumerate(KIDS_SIZES) if _KIDS_HEIGHT[sz] >= h), len(KIDS_SIZES) - 1)
        candidates = KIDS_SIZES[i: i + 2]  # never size a growing child down

    stocked = [s for s in (available or []) if s in sizes]

    def evaluate(sz: str) -> List[Dict[str, Any]]:
        g = dict(garment_measurements(sz, audience, category))
        if chart and sz in chart:
            g.update(chart[sz])
        out = []
        for r in regions:
            if body[r] is None:
                continue
            ease = g[r] - float(body[r])
            out.append({"region": r, "body_cm": round(float(body[r]), 1), "garment_cm": g[r], "ease_cm": round(ease, 1),
                        "verdict": verdict(category, r, ease - (intended if chart else 0))})
        return out

    scored = []
    for sz in candidates:
        regs = evaluate(sz)
        # never recommend a size that is tight anywhere if a non-tight one exists
        scored.append((sum(_SCORE[r["verdict"]] for r in regs),
                       sum(abs(r["ease_cm"] - (intended if chart else 0) - sum(EASE[category][r["region"]][1:3]) / 2)
                           for r in regs), sz))
    recommended = min(scored)[2]
    recommended_available = not stocked or recommended in stocked
    closest = recommended
    if not recommended_available:
        # nearest stocked size to the ideal one (by position in the size run)
        closest = min(stocked, key=lambda s: abs(sizes.index(s) - sizes.index(recommended)))
    chosen = size if size in sizes else closest
    regs = evaluate(chosen)
    worst = max(regs, key=lambda r: _SCORE[r["verdict"]]) if regs else None
    summary = "Good fit" if not worst or worst["verdict"] == "good" else \
        f"{worst['verdict'].capitalize()} at the {worst['region']} ({worst['ease_cm']:+.0f} cm ease)"
    if chart and cut != "regular":
        summary += f" ({cut} cut)"
    if not recommended_available:
        summary += f" · your size {recommended} is not in stock"
    return {"supported": True, "size": chosen, "recommended_size": recommended, "sizes": sizes,
            "available_sizes": stocked or sizes, "recommended_available": recommended_available,
            "closest_available": closest, "uses_product_chart": bool(chart), "fit_type": cut,
            "regions": regs, "summary": summary}

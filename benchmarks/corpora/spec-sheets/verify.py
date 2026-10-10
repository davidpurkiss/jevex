"""Check every label in a labels file against its document's text (#212).

    python verify.py LABELS.json TEXT_DIR

prints, per document, how many values are verified, verified only through a column
header's unit, not found, and unchecked, then each value not found with its evidence.
"""

import json
import re
import sys
from pathlib import Path

CONVERT = {"ps": 0.7355, "bhp": 0.7457, "hp": 0.7457, "cv": 0.7355}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ")).lower()


def numbers_with_unit(text: str, units: str) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    for m in re.finditer(rf"(\d{{1,4}}(?:[.,]\d+)?)\s*({units})\b", text, re.I):
        out.append((float(m.group(1).replace(",", ".")), m.group(2).lower()))
    return out


def has_number(text: str, value: float) -> bool:
    v = f"{value:g}"
    return re.search(rf"(?<![\d.,]){re.escape(v)}(?![\d]|[.,]\d)", text) is not None


def price_forms(digits: str) -> list[str]:
    n = int(digits)
    return [f"{n:,}", f"{n}", f"{n:,}".replace(",", " "), f"{n:,}".replace(",", "\u00a0")]


def header_unit_power(text: str, kw: float) -> bool:
    """A power figure whose unit is only in a column header: a "kW / ..." header with the
    value first in an "a / b" pair, or a "ps" header with a bare number converting to it.
    Weaker than the checks above, so reported separately."""
    if re.search(r"\(\s*kW\s*/", text) and re.search(rf"(?<![\d.]){kw:g}\s*/\s*\d{{2,3}}\b", text):
        return True
    if re.search(r"\(\s*ps\b|\bps\^", text, re.I):
        return any(
            abs(round(int(n) * CONVERT["ps"], 1) - kw) < 0.06
            for n in re.findall(r"(?<![\d.,])(\d{2,3})(?![\d.,])", text)
        )
    return False


def check(field: str, value: object, text: str, low: str) -> bool | str | None:
    """True if the text supports the value, False if not, None if it can't be checked."""
    if field in ("fuel_type", "automatic"):
        return None
    if field in ("make", "model", "trim"):
        return norm(str(value)) in low
    if field == "price_gbp":
        return any(form in text for form in price_forms(str(value)))
    if field == "power_kw":
        kw = float(str(value))
        if any(abs(v - kw) < 0.06 for v, _ in numbers_with_unit(text, "kW")):
            return True
        # "hp" is metric when it's DIN hp, imperial otherwise: accept either reading.
        if any(
            abs(round(v * factor, 1) - kw) < 0.06
            for v, u in numbers_with_unit(text, "PS|bhp|hp|cv")
            for factor in ({CONVERT[u], CONVERT["ps"]} if u == "hp" else {CONVERT[u]})
        ):
            return True
        # A "PS/kW" table header leaves bare pairs ("63 / 46"): accept a kW figure paired
        # with a PS figure that converts to it.
        # "PS/kW" headers leave bare pairs ("63 / 46"), "kW/PS" ones "170 (231)": accept
        # a kW figure paired with a PS figure that converts to it.
        pairs = [(ps, k) for ps, k in re.findall(r"(\d{2,3})\s*/\s*(\d{2,3}(?:\.\d)?)\b", text)] + [
            (ps, k) for k, ps in re.findall(r"\b(\d{2,3}(?:\.\d)?)\s*\((\d{2,3})\)", text)
        ]
        # ... and "hp (kW)" ones "131 (96)": the bracketed figure is the kW.
        pairs += re.findall(r"\b(\d{2,3})\s*\((\d{2,3}(?:\.\d)?)\)", text)
        if re.search(r"kW", text) is not None and any(
            abs(float(ps) * CONVERT["ps"] - float(k)) < 1.0 and abs(float(k) - kw) < 0.06
            for ps, k in pairs
        ):
            return True
        return "header" if header_unit_power(text, kw) else False
    if field in ("engine_size_cc", "co2_g_km", "top_speed_mph", "seats", "zero_to_62_s"):
        return has_number(text, float(str(value)))
    return None


def main() -> None:
    labels = json.loads(Path(sys.argv[1]).read_text())
    texts = Path(sys.argv[2])
    pages = labels.get("pages", [labels])
    total = {"ok": 0, "header": 0, "bad": 0, "unchecked": 0}
    for page in pages:
        text = (texts / (page["path"] + ".txt")).read_text()
        low = norm(text)
        bad: list[str] = []
        counts = {"ok": 0, "header": 0, "bad": 0, "unchecked": 0}
        for rec in page["records"]:
            for field, value in rec["values"].items():
                result = check(field, value, text, low)
                key = (
                    "unchecked"
                    if result is None
                    else "header"
                    if result == "header"
                    else "ok"
                    if result
                    else "bad"
                )
                counts[key] += 1
                if result is False:
                    ev = rec.get("evidence", {}).get(field, "")
                    bad.append(f"  {rec.get('entity', '?')}: {field}={value!r}  [{ev}]")
        for k in total:
            total[k] += counts[k]
        print(f"{page['path']}: {len(page['records'])} records, {counts}")
        for line in bad:
            print(line)
    print("TOTAL", total)


if __name__ == "__main__":
    main()

# Labelling the spec-sheets corpus

The answer key (`truth.json`) for jevex's real-document benchmark (#212). Each document
gets the `VehicleSpec` records a perfect extractor would return, so the benchmark is only
as good as these labels. **Label only what the document states. Never use outside
knowledge, never guess, and never fill a gap from a similar variant.**

## One record per priced variant

A record is one variant the document prices: **trim × powertrain × gearbox**. Add body
style, battery or drive when the document prices them separately (hatch and estate;
standard and long range; FWD and AWD), and model year when the document prices two (each year's variant is its own
record, with the year in its `entity`). Every such variant the document lists with a price
is a record. A document that prices only per trim ("SE L from £x", no engine named) gets
one record per trim.

Don't make records for options, packs, paint, accessories, finance or lease offers,
Motability prices, commercial, van or "Cargo" versions, or variants named without a price. The
document's own layout decides between a variant and an option: a row of its main price
table is a variant, even one named for a pack ("ST with Handling Pack"), while rows it
presents as options or option codes aren't. In prose, a gearbox or engine with its own
price is a variant even when it's called "an option".

A document's statement that clearly applies to a set of variants counts for each of them.
For example, "All models: 5 seats", or a table's column for that engine. A statement about
an engine (its fuel, cc, power) applies to every variant the document gives that same
engine, with any gearbox. A statement about some other variant doesn't.

## Fields

Leave a field out when the document doesn't state it for that variant.

| Field | How to label it |
| --- | --- |
| `make` | The brand as printed in the document's **text**, with its diacritics ("Škoda" if the text writes it so). A brand shown only as a logo image is left out: jevex reads text, so it isn't extractable. A mention anywhere in the text counts, a web address included |
| `model` | The model name only: no make, no trim, no body style, no marketing words such as "New" or "All-new" ("Focus", "Octavia", "e-3008"). An "N" or "GR" performance model whose name the document treats as the model ("Kona N") keeps it, and so does a powertrain word the document writes as part of the name ("Corsa Electric") |
| `trim` | The series or grade name as the main price table writes it, without engine or gearbox, and without fitted equipment in brackets or after "with": "Excel" for "Excel (Pan Roof)", "ST" for "ST with Handling Pack". A name the table gives as a series of its own stays whole ("ST-Line X Black Package") |
| `fuel_type` | `petrol`, `diesel`, `hybrid` (full, self-charging), `phev` (plug-in) or `ev`. A mild hybrid (MHEV, 48V) is its fuel: `petrol` or `diesel`. The document's own words decide: a powertrain it calls "Hybrid" is `hybrid` unless it says mild hybrid or 48V, whatever outside knowledge says. Left out if the document never names the fuel (a "1.0T" engine with no "petrol" anywhere), even if it's obvious |
| `engine_size_cc` | Only when stated in cc (or a cc value in a spec table). A litre figure alone ("1.5 TSI") isn't converted. Leave it out for EVs |
| `power_kw` | kW as stated. Otherwise convert: PS, DIN hp or CV (metric horsepower) × 0.7355; bhp × 0.7457; rounded to one decimal. A plain "hp" marked neither DIN nor bhp is ambiguous (makers use it for both, and the two differ by more than eval's tolerance), so leave power out unless the document also gives kW, PS or bhp. For a full hybrid or plug-in, the system output if the document gives it, else its only stated power figure; with separate engine and motor figures but no system output, leave it out. A mild hybrid takes its engine's figure |
| `zero_to_62_s` | The 0–62 mph time in seconds. A 0–100 km/h time isn't the same, so leave it out |
| `top_speed_mph` | Top speed in mph (not km/h). Several figures for one variant (per drive mode) aren't one value, so leave it out |
| `co2_g_km` | The variant's combined CO₂ in g/km, as the document gives it (the WLTP figure where several are given). A figure marked provisional or not labelled WLTP still counts; say so in `notes`. A range ("120–125") isn't one value, so leave it out. An EV's 0 only if stated |
| `price_gbp` | A string of digits ("27495"): the variant's on-the-road (OTR) price. Without an OTR price, the list price the document gives for the variant (MRRP, "from"); say so in the page's `notes` |
| `seats` | Number of seats, if stated for the variant or the model |
| `automatic` | `false` only where the document says manual (a row that's merely not the DSG one isn't enough). `true` for automatic, DCT/DSG, CVT/e-CVT, AMT (automated manual), a named automatic or dual-clutch gearbox (for example PowerShift, S tronic, Xtronic, EAT8, EDC, e-DSC), or an EV the document calls automatic; `false` for manual. Left out if the document doesn't say |

## Format

One entry per document in `truth.json`'s `pages`:

```json
{"path": "ford-focus-pricelist-2026.pdf", "schema": "VehicleSpec", "locale": "en-GB",
 "notes": "Prices are the p3 OTR column. Power is kW as stated (p8, p9).",
 "records": [
   {"entity": "ST-Line 5 door 1.0L EcoBoost 125PS mHEV 6 Speed Manual",
    "values": {"make": "Ford", "model": "Focus", "trim": "ST-Line", "fuel_type": "petrol",
               "power_kw": 92, "zero_to_62_s": 10.2, "top_speed_mph": 124,
               "co2_g_km": 119, "price_gbp": "29575", "automatic": false},
    "evidence": {"power_kw": "p8: 125PS (92kW)",
                 "price_gbp": "p3: ST-Line 5 door 1.0L EcoBoost 125PS mHEV … £29,575.00",
                 "automatic": "p3 Transmission: 6 Speed Manual"}}
 ]}
```

When the document contradicts itself (a power figure that differs between two pages), use
the figure from the variant's own specification or price row, and note the conflict in
`notes`. Within a row, a figure in the column for that value beats a number in the
variant's name ("mild hybrid 140" in a row whose power column says 130). If the figures
still disagree and neither rule settles it, leave the field out and note why.

This is one of the corpus's own records (`truth.json` gives `evidence` for every value).
`entity` is a readable label for the variant; eval uses it only to break ties. `evidence`
says where each value came from: a page (for PDFs) and the words it's read from. It's for
checking, not scoring.

## Checking labels

`verify.py` checks every labelled value against its document's text:
- names appear in it;
- prices appear with their thousands separators;
- numbers appear;
- kW matches a stated kW, PS or bhp figure after conversion, or a "PS/kW" pair.

It proves a value is in the document, not that it belongs to the right variant, so a
person spot-checks a random sample (#212). It can't check `fuel_type` or `automatic`.
Values it accepts only because a column header gives the unit are counted as `header`,
and go in the sample.

```sh
uv run python benchmarks/corpora/spec-sheets/extract_text.py DOCS_DIR /tmp/spec-text
uv run python benchmarks/corpora/spec-sheets/verify.py benchmarks/corpora/spec-sheets/truth.json /tmp/spec-text
```

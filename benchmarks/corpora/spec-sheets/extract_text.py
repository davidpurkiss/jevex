"""Write each corpus document's text, as jevex's document gate reads it, for the label
checker (#212).

    uv run python benchmarks/corpora/spec-sheets/extract_text.py DOCS_DIR TEXT_DIR

PDFs give their text layer with ``=== page N ===`` markers (``jevex.PdfTextReader``), and
HTML pages their text (``jevex.HtmlTextReader``).
"""

import sys
from pathlib import Path

from jevex import Document, HtmlTextReader, PdfTextReader


def main() -> None:
    docs, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    pdf, html = PdfTextReader(), HtmlTextReader()
    for f in sorted(docs.iterdir()):
        if f.suffix not in (".pdf", ".html"):
            continue
        document = Document.from_path(f)
        read = pdf.read(document) if f.suffix == ".pdf" else html.read(document)
        if read is None:
            raise SystemExit(f"{f.name}: no text")
        if read.pages:
            text = "".join(f"\n\n=== page {i} ===\n{p}" for i, p in enumerate(read.pages, 1))
        else:
            text = read.text
        (out / f"{f.name}.txt").write_text(text)


if __name__ == "__main__":
    main()

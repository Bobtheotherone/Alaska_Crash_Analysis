"""Numeric-token diff between two PDF builds (no-drift gate, Part E steps 16-17).

Extracts the text of both PDFs, collects every numeric token (digits with optional
thousands commas and decimal part), and reports the multiset difference. The correction
release must show ONLY the authorized token changes; anything else is drift.

Usage:
    python tools/pdf_token_diff.py OLD.pdf NEW.pdf [--out diff.json]

Prints a summary and (with --out) writes the full machine-readable diff.
Exit code 0 always — the AUTHORIZATION judgment is made by the caller/report, not here.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?")


def tokens_by_page(pdf: Path) -> list[Counter]:
    from pypdf import PdfReader
    out = []
    for page in PdfReader(str(pdf)).pages:
        text = page.extract_text() or ""
        out.append(Counter(TOKEN.findall(text)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    old_pages = tokens_by_page(Path(args.old))
    new_pages = tokens_by_page(Path(args.new))
    old_all = sum(old_pages, Counter())
    new_all = sum(new_pages, Counter())

    added = new_all - old_all
    removed = old_all - new_all

    def pages_with(tok: str, pages: list[Counter]) -> list[int]:
        return [i + 1 for i, c in enumerate(pages) if tok in c]

    result = {
        "old": str(args.old), "new": str(args.new),
        "old_pages": len(old_pages), "new_pages": len(new_pages),
        "old_total_tokens": sum(old_all.values()), "new_total_tokens": sum(new_all.values()),
        "added": {t: {"count": c, "pages": pages_with(t, new_pages)}
                  for t, c in sorted(added.items())},
        "removed": {t: {"count": c, "pages": pages_with(t, old_pages)}
                    for t, c in sorted(removed.items())},
        "added_token_set": sorted(added),
        "removed_token_set": sorted(removed),
    }
    print(f"pages: {len(old_pages)} -> {len(new_pages)}")
    print(f"added tokens   ({len(added)} distinct): {sorted(added)}")
    print(f"removed tokens ({len(removed)} distinct): {sorted(removed)}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

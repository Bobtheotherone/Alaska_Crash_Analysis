"""Positional LaTeX result-table parsing and cell verification.

Replaces the release verifier's row-substring matching (audit finding N2, 2026-07-15):
matching an expected value as a substring of a whole concatenated row can false-pass a
correct value in the wrong column — exactly how the ``tab:prior-performance`` decision-tree
macro-F1 misrounding (0.452 printed, exact 0.451) survived while the row's oMAE cell held
0.451.

This module parses each labeled ``table``/``longtable`` environment into
``(row_key, [cell, ...])`` records and verifies a spec of
``(table_id, row_key, column_index, expected_formatted_value)`` entries with exact,
per-cell string equality. It rejects:

* a correct value in the wrong column (cells are compared positionally);
* swapped columns (both positions mismatch);
* duplicate row keys inside one table;
* missing or extra rows (when ``exact_rows``) and wrong per-row column counts;
* header drift (declared fragments must appear in the declared header cell).

Staleness of generated table artifacts is checked by the caller (regenerate-and-compare).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["ParsedTable", "TableSpec", "normalize_cell", "parse_table", "verify_table"]

_SKIP_PREFIXES = (
    "%", "\\midrule", "\\bottomrule", "\\toprule", "\\multicolumn", "\\end", "\\tabnote",
    "\\begin", "\\caption", "\\label", "\\item", "\\rule", "\\endfirsthead", "\\endhead",
    "\\endfoot", "\\endlastfoot", "\\cmidrule",
)


def normalize_cell(raw: str) -> str:
    """Reduce a LaTeX table cell to comparable plain text.

    Drops ``\\textsuperscript{...}`` (marker plus argument), unwraps every other
    ``\\command`` token (argument kept), removes math ``$``, braces, and table spacing,
    and collapses whitespace. ``---`` (the undefined-value em dash) survives literally.
    """
    s = raw.strip()
    s = re.sub(r"\\textsuperscript\s*\{[^{}]*\}", "", s)
    s = s.replace(r"\\", " ")  # \makecell line breaks
    s = re.sub(r"\\[A-Za-z]+\s*", " ", s)
    s = s.replace("$", "").replace("{", "").replace("}", "")
    s = s.replace("~", " ")
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class ParsedTable:
    label: str
    header: List[str]
    rows: List[Tuple[str, List[str]]]  # (normalized row key, all normalized cells incl. col 0)

    def row_keys(self) -> List[str]:
        return [k for k, _ in self.rows]


@dataclass
class TableSpec:
    label: str
    n_cols: int
    header: Dict[int, str] = field(default_factory=dict)   # column index -> required fragment
    rows: Dict[str, Dict[int, str]] = field(default_factory=dict)  # row key -> {col -> exact value}
    exact_rows: bool = True


def _find_env_block(tex: str, label: str) -> Optional[str]:
    pos = tex.find("\\label{%s}" % label)
    if pos < 0:
        return None
    starts = [(tex.rfind("\\begin{%s}" % env, 0, pos), env) for env in ("table", "longtable")]
    start, env = max(starts)
    if start < 0:
        return None
    end = tex.find("\\end{%s}" % env, pos)
    if end < 0:
        return None
    return tex[start:end]


def _logical_rows(block_lines: Sequence[str]) -> List[str]:
    """Data lines: contain a column separator, are not structural commands."""
    out = []
    for ln in block_lines:
        t = ln.strip()
        if not t or t.startswith(_SKIP_PREFIXES):
            continue
        if "&" not in t:
            continue
        out.append(t.rstrip("\\").strip() if t.endswith("\\\\") else t)
    return out


def _split_cells(row: str) -> List[str]:
    row = row.strip()
    if row.endswith("\\\\"):
        row = row[:-2]
    return [normalize_cell(c) for c in row.split("&")]


def parse_table(tex: str, label: str) -> Optional[ParsedTable]:
    block = _find_env_block(tex, label)
    if block is None:
        return None
    lines = block.splitlines()

    # Header: first '&' line between \toprule and the first \midrule.
    header: List[str] = []
    seen_top = False
    for ln in lines:
        t = ln.strip()
        if "\\toprule" in t:
            seen_top = True
            continue
        if seen_top and "\\midrule" in t:
            break
        if seen_top and "&" in t and not t.startswith("\\multicolumn"):
            header = _split_cells(t)
            break

    # Data region.
    if "\\endlastfoot" in block:
        region = block.split("\\endlastfoot", 1)[1]
    else:
        i = block.find("\\midrule")
        region = block[i + len("\\midrule"):] if i >= 0 else block
        j = region.find("\\bottomrule")
        if j >= 0:
            region = region[:j]

    rows: List[Tuple[str, List[str]]] = []
    for raw in _logical_rows(region.splitlines()):
        cells = _split_cells(raw)
        if cells and cells[0]:
            rows.append((cells[0], cells))
    return ParsedTable(label=label, header=header, rows=rows)


def verify_table(tex: str, spec: TableSpec) -> List[Tuple[str, bool, str]]:
    """Return (check_name, ok, detail) triples for every structural and cell check."""
    results: List[Tuple[str, bool, str]] = []
    parsed = parse_table(tex, spec.label)
    if parsed is None:
        return [(f"{spec.label}: table found", False, "no such labeled environment")]
    results.append((f"{spec.label}: table found", True, ""))

    for idx, frag in spec.header.items():
        ok = idx < len(parsed.header) and frag in parsed.header[idx]
        results.append((f"{spec.label}: header[{idx}] contains {frag!r}", ok,
                        parsed.header[idx] if idx < len(parsed.header) else "<absent>"))

    keys = parsed.row_keys()
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    results.append((f"{spec.label}: no duplicate row keys", not dupes, str(dupes)))

    missing = [k for k in spec.rows if k not in keys]
    results.append((f"{spec.label}: all expected rows present", not missing, str(missing)))
    if spec.exact_rows:
        extra = [k for k in keys if k not in spec.rows]
        results.append((f"{spec.label}: no unexpected rows", not extra, str(extra)))

    by_key = {}
    for k, cells in parsed.rows:
        by_key.setdefault(k, cells)
    for key, cols in spec.rows.items():
        cells = by_key.get(key)
        if cells is None:
            continue  # covered by the missing-rows check
        results.append((f"{spec.label} [{key}]: column count == {spec.n_cols}",
                        len(cells) == spec.n_cols, f"got {len(cells)}: {cells}"))
        for idx, want in cols.items():
            got = cells[idx] if idx < len(cells) else "<absent>"
            results.append((f"{spec.label} [{key}] col {idx}: == {want!r}",
                            got == want, f"got {got!r}"))
    return results

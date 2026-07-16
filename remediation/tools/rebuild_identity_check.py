"""R-006: source-rebuild identity check, distinct from release-copy identity.

Copies the committed LaTeX source to a fresh short-path build directory, rebuilds the
PDF with the local toolchain (pdflatex x1, bibtex, pdflatex x3 — exactly build.sh), and
compares the result against the released PDF at three levels:

  * binary identity (SHA-256),
  * extracted-text identity (pypdf, page by page),
  * page count.

Writes a machine-readable result for the final verification report. PDF metadata
(CreationDate/ModDate, ID) generally differs across rebuilds, so text identity with a
different binary hash is the expected outcome and is reported as exactly that.

Usage:
    python tools/rebuild_identity_check.py --release-pdf <path> [--build-dir C:/tmp/aca-rebuild]
                                           [--out rebuild_identity.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REM = Path(__file__).resolve().parents[1]
SRC = REM / "paper" / "latex"

EXCLUDE_SUFFIX = {".aux", ".log", ".out", ".toc", ".lof", ".lot", ".blg", ".bbl", ".pdf"}
EXCLUDE_NAMES = {"page_render", "__pycache__", "figure_generation"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def copy_source(dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for p in SRC.rglob("*"):
        rel = p.relative_to(SRC)
        if any(part in EXCLUDE_NAMES for part in rel.parts):
            continue
        if p.is_dir():
            continue
        if p.suffix.lower() in EXCLUDE_SUFFIX and rel.parts[0] != "figures":
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, out)


def run(cmd, cwd) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)


def extract_text(pdf: Path) -> list[str]:
    from pypdf import PdfReader
    return [(pg.extract_text() or "") for pg in PdfReader(str(pdf)).pages]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-pdf", required=True)
    ap.add_argument("--build-dir", default="C:/tmp/aca-rebuild")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    release_pdf = Path(args.release_pdf)
    build_dir = Path(args.build_dir)
    copy_source(build_dir)

    latex_ver = run(["pdflatex", "--version"], build_dir).stdout.splitlines()[0]
    seq = [["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
           ["bibtex", "main"],
           ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
           ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
           ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]]
    for cmd in seq:
        r = run(cmd, build_dir)
        if r.returncode != 0 and cmd[0] == "pdflatex":
            print(r.stdout[-2000:])
            raise SystemExit(f"rebuild failed at: {' '.join(cmd)}")

    rebuilt = build_dir / "main.pdf"
    rel_sha, reb_sha = sha256(release_pdf), sha256(rebuilt)
    rel_text, reb_text = extract_text(release_pdf), extract_text(rebuilt)
    text_identical = rel_text == reb_text
    diff_pages = [i + 1 for i, (a, b) in enumerate(zip(rel_text, reb_text)) if a != b]

    result = {
        "toolchain": latex_ver,
        "build_sequence": "pdflatex x1, bibtex, pdflatex x3 (build.sh)",
        "release_pdf": {"path": str(release_pdf), "sha256": rel_sha, "pages": len(rel_text)},
        "rebuilt_pdf": {"sha256": reb_sha, "pages": len(reb_text)},
        "binary_identical": rel_sha == reb_sha,
        "extracted_text_identical": text_identical,
        "text_differing_pages": diff_pages,
        "note": "PDF metadata (CreationDate/ModDate/ID) differs across rebuilds; "
                "text identity with a different binary hash is the expected outcome",
    }
    print(json.dumps({k: result[k] for k in
                      ("binary_identical", "extracted_text_identical",
                       "text_differing_pages")}, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0 if text_identical else 1


if __name__ == "__main__":
    sys.exit(main())

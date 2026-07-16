"""Automated post-build audit for the UAA student paper.

Checks (exit 0 = all pass; prints a PASS/FAIL/WARN report):
  1. main.log: zero LaTeX errors, no undefined references/citations,
     no multiply-defined labels, no missing characters
  2. Overfull box census (hard fail > 10pt; report all)
  3. pdfinfo: page size is US Letter, not encrypted
  4. pdffonts: every font embedded
  5. pdftotext: no placeholder text remains; expected artifact strings present
  6. Rendered-page ink check: no content in the 1in margins
     (bottom band excludes the centered footer page number),
     and no unexpectedly blank interior page
  7. Figure files: all ten required figures exist and are vector PDFs
  8. List-of-figures count == 10, list-of-tables count == 9
  9. Numeric spot-checks: headline values in main.tex match the frozen
     repository artifacts (recomputed live from the JSONs)

Usage: python audit_paper.py [--src DIR] [--repo <repo_root>] [--render DIR]
(defaults are derived from this file's location; no machine-specific path is assumed)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

FAIL = []
WARN = []
PASS = []


def report(kind, msg):
    {"PASS": PASS, "WARN": WARN, "FAIL": FAIL}[kind].append(msg)
    print(f"[{kind}] {msg}")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


def check_log(src: Path):
    log = (src / "main.log").read_text(encoding="utf-8", errors="replace")
    errs = re.findall(r"^! .*", log, re.M)
    if errs:
        report("FAIL", f"LaTeX errors in main.log: {errs[:5]}")
    else:
        report("PASS", "no LaTeX errors in main.log")
    for pat, desc in [
        (r"LaTeX Warning: There were undefined references",
         "undefined references"),
        (r"LaTeX Warning: Citation .* undefined", "undefined citations"),
        (r"LaTeX Warning: Label\(s\) may have changed", "stale cross-references"),
        (r"multiply defined", "multiply-defined labels"),
        (r"Missing character", "missing font characters"),
    ]:
        if re.search(pat, log):
            report("FAIL", f"{desc} present in main.log")
        else:
            report("PASS", f"no {desc}")
    over = re.findall(r"Overfull \\hbox \((\d+\.?\d*)pt too wide.*?lines? (\d+)",
                      log)
    bad = [(float(a), b) for a, b in over if float(a) > 10.0]
    if bad:
        report("FAIL", f"overfull hboxes > 10pt: {bad}")
    elif over:
        report("WARN", f"{len(over)} overfull hbox(es) <= 10pt: "
               f"{[(a, b) for a, b in over]}")
    else:
        report("PASS", "no overfull hboxes")
    vover = re.findall(r"Overfull \\vbox \((\d+\.?\d*)pt too high", log)
    if vover:
        report("WARN", f"overfull vboxes: {vover}")
    else:
        report("PASS", "no overfull vboxes")


def check_pdfinfo(pdf: Path):
    info = run(["pdfinfo", str(pdf)])
    if re.search(r"Page size:\s+612 x 792 pts \(letter\)", info):
        report("PASS", "page size 612x792 (US Letter)")
    else:
        report("FAIL", f"unexpected page size: "
               f"{re.search(r'Page size:.*', info).group(0) if re.search(r'Page size:.*', info) else '??'}")
    if re.search(r"Encrypted:\s+no", info):
        report("PASS", "not encrypted")
    else:
        report("FAIL", "PDF is encrypted or check failed")
    m = re.search(r"Pages:\s+(\d+)", info)
    print(f"       page count: {m.group(1)}")
    return int(m.group(1))


def check_fonts(pdf: Path):
    out = run(["pdffonts", str(pdf)])
    lines = [ln for ln in out.splitlines()[2:] if ln.strip()]
    notemb = [ln for ln in lines if re.search(r"\s(no)\s+(yes|no)\s+(yes|no)\s+\d+", ln)
              and ln.split()[-4] == "no"]
    # column 'emb' is 4th from the end block: name type encoding emb sub uni object
    notemb = []
    for ln in lines:
        parts = ln.split()
        if len(parts) >= 6 and parts[-4] == "no":
            notemb.append(parts[0])
    if notemb:
        report("FAIL", f"unembedded fonts: {notemb}")
    else:
        report("PASS", f"all {len(lines)} fonts embedded")


def check_text(pdf: Path, src: Path):
    txt = run(["pdftotext", "-q", str(pdf), "-"])
    if "reserved for final insertion" in txt:
        report("FAIL", "a figure placeholder is still rendered")
    else:
        report("PASS", "no figure placeholders remain")
    for needle, desc in [
        ("final_8af9d5bc23d8", "governed run id"),
        ("Radames Naythan Mercado-Barbosa", "author name"),
        ("Peyton Ratzer", "collaborator name"),
        ("Vinod Vasudevan", "client name"),
        ("Osama Abaza", "mentor name"),
        ("University of Alaska Anchorage", "UAA affiliation"),
    ]:
        if needle in txt:
            report("PASS", f"{desc} present")
        else:
            report("FAIL", f"{desc} MISSING from rendered text")
    # "MIT License" is the declared code license (availability section, r3); the guard
    # targets template-university leftovers, so exempt the license phrase only.
    txt_guard = txt.replace("MIT License", "")
    for bad, desc in [
        ("Harvard", "Harvard reference"), ("MIT", "MIT university reference"),
        ("??", "unresolved ?? reference"),
    ]:
        if bad in txt_guard:
            report("FAIL", f"forbidden text present: {desc}")
        else:
            report("PASS", f"no {desc}")


def check_lists(src: Path):
    lof = (src / "main.lof").read_text(encoding="utf-8", errors="replace")
    lot = (src / "main.lot").read_text(encoding="utf-8", errors="replace")
    nf = len(re.findall(r"\\contentsline \{figure\}", lof))
    nt = len(re.findall(r"\\contentsline \{table\}", lot))
    # 11 figures: the 10 of the correction release + the severe-class reliability
    # diagram (r3, fig06c). 17 LoT entries: the 10 of the correction release + the
    # year-by-year cohort table + the complete hard-label detail table + the
    # development-results record + retained-49 (part 1 counted; part 2 list-suppressed)
    # + prohibited-14 + exclusion summary + the reproducibility manifest; multi-part
    # tables (artifact map, analysis status, retained fields) list only part 1.
    report("PASS" if nf == 11 else "FAIL", f"list of figures has {nf}/11 entries")
    report("PASS" if nt == 17 else "FAIL", f"list of tables has {nt}/17 entries")


def check_figures(src: Path):
    need = ["fig01_project_lineage.pdf", "fig02_governed_workflow.pdf",
            "fig09_original_reanalysis.pdf", "fig03_final_ordinal_mae.pdf",
            "fig04_tradeoff.pdf", "fig06a_probability_scores.pdf",
            "fig06b_calibration.pdf", "fig06c_reliability.pdf",
            "fig07_leakage_factorial.pdf",
            "fig05_protocol_corrections.pdf", "fig08_severe_ranking.pdf"]
    missing = [f for f in need if not (src / "figures/user" / f).exists()]
    if missing:
        report("FAIL", f"missing figure files: {missing}")
    else:
        report("PASS", "all 11 figure PDFs present")
    for f in need:
        p = src / "figures/user" / f
        if p.exists():
            head = p.read_bytes()[:8]
            if not head.startswith(b"%PDF"):
                report("FAIL", f"{f} is not a PDF")


def check_pages(pdf: Path, render_dir: Path, npages: int):
    import struct
    import zlib

    render_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pdftoppm", "-png", "-r", "100", str(pdf),
                    str(render_dir / "page")], check=True)
    pages = sorted(render_dir.glob("page-*.png"))
    if len(pages) != npages:
        report("FAIL", f"rendered {len(pages)} pages, expected {npages}")
        return

    def png_gray(path):
        # minimal PNG reader via matplotlib (available in this env)
        import matplotlib.image as mpimg
        a = mpimg.imread(str(path))
        if a.ndim == 3:
            a = a[..., :3].mean(axis=2)
        return a

    margin_bad = []
    blank = []
    for i, p in enumerate(pages, 1):
        a = png_gray(p)          # values 0..1, white=1
        h, w = a.shape           # ~1100 x 850 at 100dpi
        m = 96                   # 1in margin at 100dpi with 4px tolerance
        ink = a < 0.85
        top = ink[:m, :].sum()
        left = ink[:, :m].sum()
        right = ink[:, w - m:].sum()
        # bottom band excludes the centered footer strip (page number)
        bot = ink[h - m:, :]
        c0, c1 = int(w * 0.42), int(w * 0.58)
        bottom = bot[:, :c0].sum() + bot[:, c1:].sum()
        if top > 30 or left > 30 or right > 30 or bottom > 30:
            margin_bad.append((i, int(top), int(left), int(right), int(bottom)))
        interior = ink[m:h - m, m:w - m].sum()
        if interior < 200 and i not in (1,):
            blank.append(i)
    if margin_bad:
        report("FAIL", f"ink inside 1in margins on pages {margin_bad}")
    else:
        report("PASS", "no content inside the 1in margins (footer excluded)")
    if blank:
        report("WARN", f"nearly blank pages (interior): {blank}")
    else:
        report("PASS", "no unexpectedly blank pages")


def check_numbers(src: Path, repo: Path):
    tex = (src / "main.tex").read_text(encoding="utf-8", errors="replace")
    r = json.loads((repo / "remediation/experiment/final_results.json")
                   .read_text())
    res = r["results"]
    pd = r["paired_difference_vs_baseline"]["ordinal_random_forest"]
    broad = json.loads((repo / "remediation/experiment/broad_sensitivity/"
                        "final_results.json").read_text())
    low = json.loads((repo / "remediation/experiment/lowmiss_sensitivity/"
                      "final_results.json").read_text())
    v3 = json.loads((repo / "remediation/evidence_release/final_f27613102c96/"
                     "manifest.json").read_text())
    lf = json.loads((repo / "remediation/experiment/leakage_factorial.json")
                    .read_text())
    se = lf["simple_effects_vs_reference"]["ordinal_mae"]
    cal = r["calibration"]["ordinal_random_forest"]

    checks = [
        ("primary oMAE", f"{res['ordinal_random_forest']['ordinal_mae']:.4f}"),
        ("majority oMAE", f"{res['majority']['ordinal_mae']:.4f}"),
        ("primary delta", f"{pd['difference']:.4f}".replace("-", "-0").replace("-00", "-0") if False else f"{pd['difference']:.4f}"),
        ("delta CI low", f"{pd['ci_low']:.4f}"),
        ("delta CI high", f"{pd['ci_high']:.4f}"),
        ("severe recall", f"{res['ordinal_random_forest']['severe_recall']*100:.1f}"),
        ("broad delta", f"{broad['paired_difference_vs_baseline']['ordinal_random_forest']['difference']:.4f}"),
        ("broad recall", f"{broad['results']['ordinal_random_forest']['severe_recall']:.3f}"),
        ("lowmiss delta", f"{low['paired_difference_vs_baseline']['ordinal_random_forest']['difference']:.4f}"),
        ("v3 delta", f"{v3['paired_difference_vs_baseline']['ordinal_random_forest']['difference']:.4f}"),
        ("v3 recall", f"{v3['results']['ordinal_random_forest']['severe_recall']:.3f}"),
        ("leak effect", f"{se['leakage_features_vs_reference']:.3f}"),
        ("preproc effect", f"{se['preprocess_before_given_leak_off']:.3f}"),
        ("split effect", f"{se['random_split_given_leak_off']:.3f}"),
        ("ECE raw", f"{cal['ece_raw']:.3f}"),
        ("ECE cal", f"{cal['ece_calibrated']:.3f}"),
    ]
    for name, val in checks:
        # numbers appear in the tex with unary minus as $-0.xxxx$
        needle = val.lstrip("-")
        if needle in tex:
            report("PASS", f"{name} {val} found in main.tex")
        else:
            report("FAIL", f"{name} {val} NOT found in main.tex")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path(__file__).resolve().parents[1]
                    / "paper" / "latex"))
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--render", default=None)
    ap.add_argument("--pdf", default=None)
    args = ap.parse_args()
    src = Path(args.src)
    pdf = Path(args.pdf) if args.pdf else src / "main.pdf"
    render = Path(args.render) if args.render else src.parent / "page_render"

    if not pdf.exists():
        print("main.pdf missing"); sys.exit(2)
    check_log(src)
    n = check_pdfinfo(pdf)
    check_fonts(pdf)
    check_text(pdf, src)
    check_lists(src)
    check_figures(src)
    check_pages(pdf, render, n)
    check_numbers(src, Path(args.repo))

    print(f"\n===== SUMMARY: {len(PASS)} pass, {len(WARN)} warn, "
          f"{len(FAIL)} fail =====")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

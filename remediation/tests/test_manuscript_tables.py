"""Regression tests for positional manuscript-table verification (audit finding N2).

The 2026-07-15 AXIOM-LOCAL audit of the f8aab323 build found that the release checker
matched expected table values as substrings of the concatenated row text, so a correct
value in ANY column satisfied the check for EVERY column. That is exactly how the
``tab:prior-performance`` decision-tree macro-F1 misrounding (printed 0.452, exact
0.451490 -> 0.451) survived: the row's oMAE cell held 0.451, so the substring check for
the macro-F1 cell false-passed. These tests reproduce that mechanism and pin the
positional comparator's rejection behavior.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crashsev.manuscript_tables import (  # noqa: E402
    TableSpec, normalize_cell, parse_table, verify_table,
)


def mini_tex(mf1="0.451", omae="0.451", extra_row="", drop_last_col=False):
    last = "" if drop_last_col else " & 0.472"
    return (
        "\\begin{table}[H]\n"
        "\\caption{Mini.}\n"
        "\\label{tab:prior-performance}\n"
        "\\begin{tabularx}{\\textwidth}{@{}Yrrrrr@{}}\n"
        "\\toprule\n"
        "\\textbf{Model} & \\textbf{Acc.} & \\textbf{oMAE} & \\textbf{QWK} & "
        "\\textbf{Macro-F1} & \\textbf{Bal. acc.} \\\\\n"
        "\\midrule\n"
        f"Decision tree & 0.580 & {omae} & 0.221 & {mf1}{last} \\\\\n"
        f"{extra_row}"
        "\\bottomrule\n"
        "\\end{tabularx}\n"
        "\\end{table}\n"
    )


SPEC = TableSpec(
    label="tab:prior-performance",
    n_cols=6,
    header={0: "Model", 1: "Acc.", 2: "oMAE", 3: "QWK", 4: "Macro-F1", 5: "Bal. acc."},
    rows={"Decision tree": {1: "0.580", 2: "0.451", 3: "0.221", 4: "0.451", 5: "0.472"}},
)


def failures(tex, spec=SPEC):
    return [name for name, ok, _ in verify_table(tex, spec) if not ok]


def test_correct_table_passes():
    assert failures(mini_tex()) == []


def test_shipped_false_pass_mechanism_is_rejected():
    """The exact defect: macro-F1 printed 0.452 while the oMAE column holds 0.451."""
    tex = mini_tex(mf1="0.452")
    # The pre-hardening mechanism (expected value as substring of the joined row)
    # accepts this row, because 0.451 appears in the oMAE column:
    row = "Decision tree & 0.580 & 0.451 & 0.221 & 0.452 & 0.472"
    assert "0.451" in row  # the blind spot this module exists to close
    # The positional comparator rejects it, at the right cell:
    fails = failures(tex)
    assert any("col 4" in f for f in fails), fails
    assert not any("col 2" in f for f in fails)


def test_right_value_wrong_column_rejected():
    """Swapped columns: both affected cells must fail."""
    tex = mini_tex().replace("& 0.451 & 0.221 &", "& 0.221 & 0.451 &")
    fails = failures(tex)
    assert any("col 2" in f for f in fails), fails
    assert any("col 3" in f for f in fails), fails


def test_duplicate_row_key_rejected():
    tex = mini_tex(extra_row="Decision tree & 0.580 & 0.451 & 0.221 & 0.451 & 0.472 \\\\\n")
    assert any("duplicate" in f for f in failures(tex))


def test_missing_column_rejected():
    assert any("column count" in f for f in failures(mini_tex(drop_last_col=True)))


def test_unexpected_row_rejected():
    tex = mini_tex(extra_row="Mystery model & 0.1 & 0.2 & 0.3 & 0.4 & 0.5 \\\\\n")
    assert any("no unexpected rows" in f for f in failures(tex))


def test_missing_table_rejected():
    assert any("table found" in f for f in failures("\\section{No tables here}"))


def test_missing_row_rejected():
    spec = TableSpec(label=SPEC.label, n_cols=6, header=SPEC.header,
                     rows={**SPEC.rows, "XGBoost": {1: "0.627"}})
    assert any("all expected rows present" in f for f in failures(mini_tex(), spec))


def test_normalize_cell_strips_formatting():
    assert normalize_cell("\\textbf{$-0.0144$ [$-0.0231,-0.0063$]}") == "-0.0144 [-0.0231,-0.0063]"
    assert normalize_cell("1.000\\textsuperscript{a}") == "1.000"
    assert normalize_cell("\\bfseries 0.3469") == "0.3469"
    assert normalize_cell("{\\textbf{\\makecell{Ordinal\\\\error}}}") == "Ordinal error"
    assert normalize_cell("---") == "---"
    assert normalize_cell("Proportional-odds logit\\textsuperscript{b}") == "Proportional-odds logit"


def test_longtable_endlastfoot_rows_parsed():
    tex = (
        "\\begin{longtable}{@{}Lr@{}}\n"
        "\\caption{Mini long.}\\label{tab:mini-long}\\\\\n"
        "\\toprule\n"
        "\\textbf{Model} & \\textbf{oMAE} \\\\\n"
        "\\midrule\n"
        "\\endfirsthead\n"
        "\\multicolumn{2}{l}{continued}\\\\\n"
        "\\toprule\n"
        "\\textbf{Model} & \\textbf{oMAE} \\\\\n"
        "\\midrule\n"
        "\\endhead\n"
        "\\midrule\n"
        "\\multicolumn{2}{r}{Continued on next page}\\\\\n"
        "\\endfoot\n"
        "\\bottomrule\n"
        "\\multicolumn{2}{@{}p{0.9\\textwidth}@{}}{\\footnotesize note text}\\\\\n"
        "\\endlastfoot\n"
        "Ordinal RF & 0.3469 \\\\\n"
        "Majority & 0.3614 \\\\\n"
        "\\end{longtable}\n"
    )
    parsed = parse_table(tex, "tab:mini-long")
    assert parsed is not None
    assert parsed.row_keys() == ["Ordinal RF", "Majority"]
    assert parsed.rows[0][1] == ["Ordinal RF", "0.3469"]
    assert parsed.header == ["Model", "oMAE"]

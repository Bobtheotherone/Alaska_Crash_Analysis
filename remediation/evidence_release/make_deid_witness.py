"""Generate the de-identification transform witness for a final run bundle (TRACE-001).

The complete run bundle under ``runs/final_<hash>/`` is retained locally only: its per-crash rows
are keyed by ``Crash Number`` (an identifier). The published evidence is a de-identified copy in
which the real id is replaced by an opaque surrogate ``T00000..`` (assignment: surrogates in
ascending order over the SORTED set of original ids; the mapping itself is discarded, never
persisted). Because the mapping is discarded, the de-identified files cannot be *reversed* — but
they can and must be *witnessed*: this script

  1. re-derives the de-identified files deterministically from the local bundle,
  2. hashes both the originals (cross-checked against ``manifest.json``'s ``artifact_sha256``)
     and the de-identified outputs,
  3. recomputes each model's ordinal MAE / accuracy from the de-identified rows and checks them
     against the manifest's recorded results (transform preserves y_true/y_pred/proba exactly),
  4. optionally byte-compares the outputs against an already-published evidence directory, and
  5. writes ``<run_id>/DEID_TRANSFORM.md`` — the committed witness linking original-hash ->
     de-identified-hash for every prediction file.

Run (from ``remediation/``):
    python evidence_release/make_deid_witness.py --src runs/final_f27613102c96 \
        --out <staging>/predictions_deidentified [--compare <published evidence dir>]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description="De-identification transform witness (TRACE-001).")
    ap.add_argument("--src", required=True, help="local run bundle dir (runs/final_<hash>)")
    ap.add_argument("--out", required=True, help="output dir for de-identified prediction CSVs")
    ap.add_argument("--compare", default=None,
                    help="optional already-published de-identified dir to byte-compare against")
    args = ap.parse_args(argv)

    src = Path(args.src); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    run_id = manifest["run_id"]
    art = manifest.get("artifact_sha256", {})

    pred_files = sorted(p for p in src.glob("predictions_*.csv"))
    if not pred_files:
        raise SystemExit(f"no prediction files under {src}")

    # 1. deterministic surrogate mapping over the union of original ids. PRIV-001 (v4.1): the
    #    v3/v4 transform assigned surrogates in SORTED original-id order, which preserves any
    #    ordering structure the source id encodes (e.g. time/agency sequence). Surrogates are now
    #    assigned under a seeded PERMUTATION of the sorted id list: still fully deterministic and
    #    re-runnable by any holder of the local bundle (the seed is documented below), but the
    #    published surrogate order no longer mirrors the source-id order. The id→surrogate
    #    mapping itself remains discarded, never persisted.
    ids = set()
    for p in pred_files:
        with open(p, newline="", encoding="utf-8") as fh:
            r = csv.reader(fh); next(r)
            for row in r:
                if row:
                    ids.add(row[0])
    import random
    PERM_SEED = 20260712
    ordered = sorted(ids)
    rng = random.Random(PERM_SEED)
    rng.shuffle(ordered)
    mapping = {cid: f"T{ix:05d}" for ix, cid in enumerate(ordered)}

    rows_md = []
    problems = []
    for p in pred_files:
        # Original hash vs the manifest's recorded artifact hash. KNOWN QUIRK (BUNDLE-CRLF-001,
        # found while building this witness): v3.x ``cli.write_bundle`` hashed the in-memory
        # LF-newline content but materialised the file via ``write_text`` (CRLF on Windows), so
        # ``artifact_sha256`` matches the file only after CRLF->LF normalisation. Both hashes are
        # recorded; the manifest match is checked on the normalised bytes. The v4 code writes the
        # exact bytes it hashes.
        raw_bytes = p.read_bytes()
        orig_hash = sha256_bytes(raw_bytes)
        lf_hash = sha256_bytes(raw_bytes.replace(b"\r\n", b"\n"))
        recorded = art.get(p.name)
        if recorded is not None and recorded not in (orig_hash, lf_hash):
            problems.append(f"{p.name}: neither raw nor LF-normalised hash matches manifest artifact_sha256")

        with open(p, newline="", encoding="utf-8") as fh:
            r = csv.reader(fh); header = next(r); rows = [row for row in r if row]
        # 2. transform: replace column 0 with the surrogate; everything else byte-preserved
        out_p = out / p.name
        with open(out_p, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(header)
            for row in rows:
                w.writerow([mapping[row[0]]] + row[1:])
        deid_hash = sha256_file(out_p)

        # 3. metric equality: oMAE and accuracy recomputed from the de-identified rows must
        #    equal the manifest's recorded results for this model (float-exact arithmetic).
        model = p.stem.replace("predictions_", "")
        yt_i = header.index("y_true"); yp_i = header.index("y_pred")
        with open(out_p, newline="", encoding="utf-8") as fh:
            r = csv.reader(fh); next(r)
            drows = [row for row in r if row]
        n = len(drows)
        omae = sum(abs(int(row[yt_i]) - int(row[yp_i])) for row in drows) / n
        acc = sum(int(row[yt_i]) == int(row[yp_i]) for row in drows) / n
        rec = manifest["results"].get(model, {})
        omae_ok = abs(omae - rec.get("ordinal_mae", float("nan"))) < 1e-12
        acc_ok = abs(acc - rec.get("accuracy", float("nan"))) < 1e-12
        if not (omae_ok and acc_ok):
            problems.append(f"{model}: de-identified metrics do not reproduce manifest results")

        # 4. optional byte-compare against the already-published copy
        published = "-"
        if args.compare:
            cand = Path(args.compare) / p.name
            if cand.exists():
                published = "byte-identical" if sha256_file(cand) == deid_hash else "**DIFFERS**"
                if published == "**DIFFERS**":
                    problems.append(f"{p.name}: published copy differs from regenerated transform")
            else:
                published = "not published"

        manifest_match = ("raw" if recorded == orig_hash
                          else "LF-normalised" if recorded == lf_hash
                          else "NO MATCH")
        rows_md.append((p.name, orig_hash, lf_hash, manifest_match, deid_hash, n, f"{omae:.6f}",
                        "ok" if (omae_ok and acc_ok) else "MISMATCH", published))

    # 5. witness document
    wit_dir = HERE / run_id
    wit_dir.mkdir(parents=True, exist_ok=True)
    self_hash = sha256_file(Path(__file__))
    lines = [
        f"# De-identification transform witness — `{run_id}` (TRACE-001)",
        "",
        f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} by "
        f"`evidence_release/make_deid_witness.py` (script SHA-256 `{self_hash}`).",
        "",
        "**Transform (v4.1, PRIV-001).** Column `group_id` (the real `Crash Number`) is replaced",
        "by an opaque surrogate: surrogates `T00000..` are assigned over a seeded PERMUTATION",
        "(seed 20260712, documented for bundle-holders' re-runs) of the sorted original-id set,",
        "identically across all prediction files (pairing across models is preserved); every",
        "other byte of every row is unchanged. Earlier releases assigned surrogates in sorted-id",
        "order, which preserved source-id ordering structure — a precautionary weakness, not a",
        "demonstrated re-identification path. The id mapping is **discarded, never persisted** —",
        "the transform is deterministic and re-runnable by anyone holding the local bundle, and",
        "*witnessable* by anyone holding this table.",
        "",
        "**Chain.** `manifest.json` records each ORIGINAL file's SHA-256 (`artifact_sha256`);",
        "this table records the matching DE-IDENTIFIED file's SHA-256; the published handoff's",
        "`04_MANIFEST_SHA256.txt` hashes the shipped copies. Original -> de-identified ->",
        "published is therefore a closed hash chain even though the originals stay local.",
        "",
        "**BUNDLE-CRLF-001 (disclosed).** The v3.x bundle writer hashed its in-memory LF content",
        "but materialised files with platform newlines (CRLF on Windows), so `artifact_sha256`",
        "matches the on-disk originals only after CRLF→LF normalisation. Both hashes are recorded",
        "below; the manifest match is evaluated on the normalised bytes. The v4 bundle writer",
        "writes the exact bytes it hashes.",
        "",
        "| prediction file | original SHA-256 (on disk) | original SHA-256 (LF-normalised) | matches manifest via | de-identified SHA-256 | rows | oMAE (recomputed) | metrics vs manifest | published copy |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, oh, lh, mm, dh, n, om, ok, pub in rows_md:
        lines.append(f"| `{name}` | `{oh}` | `{lh}` | {mm} | `{dh}` | {n} | {om} | {ok} | {pub} |")
    lines += [
        "",
        f"Surrogate domain: `T00000..T{len(mapping)-1:05d}` ({len(mapping)} crashes).",
        "Verification for a reviewer WITHOUT the local bundle: hash the shipped de-identified",
        "files and compare to column 3; recompute any metric from their rows and compare to",
        "`manifest.json -> results`. Verification WITH the local bundle: rerun this script.",
    ]
    if problems:
        lines += ["", "## PROBLEMS DETECTED", *[f"* {p}" for p in problems]]
    (wit_dir / "DEID_TRANSFORM.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[deid-witness] {len(pred_files)} files; {len(mapping)} crashes; "
          f"problems={len(problems)}")
    print(f"[deid-witness] wrote {wit_dir / 'DEID_TRANSFORM.md'}")
    if problems:
        for p in problems:
            print("  PROBLEM:", p)
        raise SystemExit(2)


if __name__ == "__main__":
    main()

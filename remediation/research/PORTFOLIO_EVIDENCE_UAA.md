# Portfolio Evidence — UAA MS AIDE alignment

This maps the project to the **official** program student learning outcomes of the University of
Alaska Anchorage **MS in Artificial Intelligence, Data Science, and Engineering (MS AIDE)**, verified
2026-07 from the UAA Academic Catalog:
https://catalog.uaa.alaska.edu/graduateprograms/coeng/ms-aide/

This is a portfolio-navigation aid, not an application essay. It points to concrete artifacts a
reviewer can open; it does **not** predict or assert admission, and it is kept out of the scientific
paper.

> **Program context (verbatim, catalog):** the MS AIDE is "collaboratively hosted by the Departments
> of Computer Science & Engineering, Electrical Engineering, and Geomatics, all housed within the
> College of Engineering." Core courses evidence the program's values: Advanced Machine Learning
> (CSCE A615), Advanced Data Mining (CSCE A662), Fundamentals of Data Science and Engineering
> (ES A603), Advanced Database Management Systems (GIS A658), and General Statistics for Data Science
> (STAT A611).

## Outcome-by-outcome evidence

### Outcome 1 (verbatim): "Utilize advanced data engineering and analytical algorithms"
| Evidence | Path |
|---|---|
| Executable data/target/feature **contracts** and schema validation | `crashsev/contracts.py`, `data/schema.json`, `data/target_mapping.yml`, `data/feature_availability_ledger.csv` |
| Reconstructible, group-safe **chronological split** with hashed id→partition table | `crashsev/splits.py` |
| Fold-local preprocessing pipeline (imputation, sparse one-hot) | `crashsev/preprocessing.py` |
| Ordinal algorithms: proportional-odds (gradient-checked), Frank–Hall ordinal RF with coherent probabilities, tree ensembles | `crashsev/models.py` |
| Immutable, hashed run bundles and provenance | `crashsev/cli.py`, `runs/…/manifest.json` |
**Applicant can explain:** the leakage-safe pipeline and the ordinal decomposition. **Limitation:** models are standard library learners; the contribution is the governed evaluation, not a new algorithm.

### Outcome 2 (verbatim): "Demonstrate graduate-level theory in data science and engineering"
| Evidence | Path |
|---|---|
| Ordinal decision theory (posterior median vs argmax under absolute loss) | `paper/final_paper.md` §2, §7.6; `experiment/decision_rule_sensitivity.md` |
| Proper scoring + calibration (log loss/Brier/RPS; dev-only calibrator; reliability) | `crashsev/metrics.py`, `crashsev/calibration.py`; paper §7.4 |
| Temporal transfer, leakage, and imbalance theory tied to primary sources | paper §2 (Shmueli; Savolainen et al.; Frank & Hall; Kaufman et al.; Guo et al.) |
| Uncertainty under a stated independence approximation | `crashsev/uncertainty.py`; paper §5.6, §10 |
**Applicant can explain:** why ordinal error, calibration, and temporal validation matter. **Limitation:** scholarly novelty is modest by design (a remediation study).

### Outcome 3 (verbatim): "Apply graduate-level data science and engineering knowledge to the research work or projects"
| Evidence | Path |
|---|---|
| A complete retrospective out-of-time study with a declared primary comparison and uncertainty | `paper/final_paper.md` §6–§9 |
| Controlled leakage factorial separating three evaluation defects | `crashsev/leakage_factorial.py`, `experiment/leakage_factorial.md` |
| Honest re-analysis of the prior study's own results | `crashsev/reanalysis.py`, `reanalysis/` |
| Claim-to-evidence traceability | `paper/CLAIM_EVIDENCE_MATRIX.md` |
**Applicant can explain:** the end-to-end study and what it does/does not establish. This directly matches the program's **Project Option** (a data-science-and-engineering project).

### Outcome 4 (verbatim): "Communicate and work effectively in a professional environment"
| Evidence | Path |
|---|---|
| A concise, claim–evidence-structured technical paper with reproducible tables/figures | `paper/final_paper.md`, `paper/final_paper.pdf` |
| Transparent issue disposition, protocol conformance, and deviations | `research/ISSUE_DISPOSITION_CURRENT.md`, `research/PROTOCOL_CONFORMANCE.md`, `research/PROTOCOL_DEVIATIONS.md` |
| Reproducibility and honest limitation reporting | `research/REPRODUCE.md`, `research/REPRODUCTION_LOG.md`, paper §10–§11 |
| Standalone auditor handoff with self-verification | handoff package + `09_VERIFY_HANDOFF.py` |
**Applicant can explain:** the reproduction routes and the limitations. **Limitation:** communication quality is best judged by a reviewer reading the paper.

## What this evidence does not do
* It does not predict or guarantee admission.
* It does not claim the applicant solely authored the AI-assisted remediation layer
  (`research/AUTHORSHIP_AND_AI_ASSISTANCE.md`).
* It does not substitute for the applicant's ability to defend the work in person.

## Note on admission requirements (verified, for the applicant's planning only)
The catalog lists, for standard admission, a baccalaureate "in engineering or a closely related
discipline," two recommendation letters, a résumé/CV, and a one-page personal statement; it states
**no** specific programming/statistics prerequisite and **no** portfolio requirement (a 3.00 GPA
minimum is stated only for the Accelerated junior-year option). This portfolio is therefore
*supporting* evidence of readiness, not a required component. Source:
https://catalog.uaa.alaska.edu/graduateprograms/coeng/ms-aide/

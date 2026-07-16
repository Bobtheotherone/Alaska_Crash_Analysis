# Prior-work data-lineage audit trail

Read-only forensic audit of a personal licensed backup (`%USERPROFILE%\Downloads\BACKUP\OLD_BACKUP`;
local username redacted) establishing the
provenance summarised in `research/DATA_PROVENANCE.md`. Paths are relative to the archive root
`...\OLD_BACKUP\2025\`. Nothing here was modified.

## Code generations

| Tag | Location | Author signal | Era |
|---|---|---|---|
| S23 pipeline | `Old_Work\AK_CrashDataAndML_Scripts_S23\Machine Learning Scripts\` | Colab paths under an S23 contributor's home (`%USERPROFILE%//Downloads//`; handle redacted) | Spring 2023 |
| F24 models + app | `Old_Work\ML_F24\capstone-main\ml_model\`; `Old_Work\akCrashData-main_F24\akCrashData-main\` | "Colab"; `CSCE A470` | Fall 2024 |
| 2025 remediation | `Alaska-Crash-Analysis-main\...`; `poster_viz\` | LibreOffice lock `rnmercado`, 14.10.2025 | 2025 |

## Q1/Q2 — producers of the two cleaned CSVs are NOT preserved

No `to_csv('cleaned_test_data...')` or `to_csv('new_test_data...')` anywhere. Only reads:
- `ML_F24\...\xgBoost\first_test_data_models\xgboost_three_classification (1).py:26` → `pd.read_csv('cleaned_test_data.csv')` (also `xgboost_seven_classification.ipynb:45`, `xgboost_five_classification (1).ipynb:45`, `binary_crash_severity_xgBoost.ipynb:162`).
- `ML_F24\...\xgBoost\second_test_data_models\xgboost_new_test_data.py:21` → `pd.read_csv('new_test_data_oct_7.csv', low_memory=False)` (also `multilevel\multilevel_model.py:198`, `multilevel\multilevel_random_forest.py:214,335,360`; DT/RF/SVM read `/content/drive/MyDrive/crash_data/new_test_data_Oct_7.csv`).
- `new_test_data_oct_7.csv` header has a leading unnamed index col → written with `to_csv(index=True)` after `reset_index`.

Documented ancestry (S23), keeps 38 Title-Case fields (a superset the CSVs down-select from):
- `AK_CrashDataAndML_Scripts_S23\...\Combine Crash Data.ipynb:13,30` read both raw xlsx → `:428` `to_pickle("Combined Crash Level Second Attempt.pkl")`.
- `...\Clean Combined Crash Data.ipynb:13` reads that pkl → writes `Cleaned Combined Crash Data 38 Fields.pkl` and `Cleaned Combined Crash Data.csv`.

Temporal columns present in the raw source but dropped from the CSVs:
- `Alaska-Crash-Analysis-main\...\data_cleaning\Column Analysis.txt:9-10` — common columns include `Year`, `Month`, `Time of Day`, `DateTime`, `Day of Month`, `Day of the Week`.
- `AK_CrashDataAndML_Scripts_S23\...\Crash Data Matched V1 and v2.txt` maps `Year||Year`, `Month||Month`, `Date||Date`, `Time of Day||Time of Day`.

## Q3 — raw crash files

- **`Crash Level 09-12 (1).xlsx`** — PRESENT at `2025\Crash Level 09-12 (1).xlsx` (25 MB). Referenced `Combine Crash Data.ipynb:13`; `Script to Clean.py:6`; `Clean Combined Crash Data.ipynb:15`.
- **`Crash Level 13-17 (1).xlsx`** — NOT present; existence proven by `AK_CrashDataAndML_Scripts_S23\...\.~lock.Crash Level 13-17 (1).xlsx#` (user `rnmercado`, 14.10.2025). Referenced `Combine Crash Data.ipynb:30`; `Script to Clean.py:5`; `Clean Combined Crash Data.ipynb:14`.
- **Combined 2009–2017** — only as derived `Cleaned Combined Crash Data 38 Fields.pkl` (26 MB, present); the `Combined Crash Level Second Attempt.pkl` intermediate is absent.

## Q4 — severity collapse (prior F24, 3-class) — matches this study

`xgBoost\first_test_data_models\xgboost_three_classification (1).py:27-36`:
```python
def map_severity(severity):
    if severity in ['No Apparent Injury']:            return 0
    elif severity in ['Suspected Minor Injury', 'Possible Injury']: return 1
    else:  # 'Suspected Serious Injury', 'Fatal Injury (Killed)'   return 2
df_filtered = df[~df['severity'].isin(['Unknown', 'Died Prior to Crash'])].copy()
```
Severity in `cleaned_test_data.csv` is fully non-null MMUCC text (7 categories); there is **no
blank/PDO class** in the CSV — class 0 is the explicit "No Apparent Injury". Binary/5/7-class
variants differ (`binary_crash_severity_xgBoost.ipynb:165`, `xgboost_five_classification (1).ipynb:175-179`, `xgboost_seven_classification.ipynb:169`). Django app label-encodes the chosen target generically (`akCrashData-main\crashdata\view_functions\run_ml_view.py:102-108`).

## Q5 — splits (all random, model-dependent fractions, random_state=42)

| Model | Path:line | Split |
|---|---|---|
| XGBoost (both CSVs) | `xgboost_three_classification (1).py:71`; `xgboost_new_test_data.py:76` | `test_size=0.2, random_state=42` (+`SMOTE(random_state=42)`) |
| Decision Tree | `decisionTree\crash_decision_tree.py:101` | `test_size=0.3, random_state=42` |
| Random Forest | `randomForest\crash_random_forest.py:96` | `test_size=0.3, random_state=42` |
| SVM | `svm\crash_svm.py:102` | `test_size=0.3, random_state=42` |
| Multilevel | `multilevel\multilevel_model.py:72`; `multilevel_random_forest.py:172` | `KFold(5, shuffle=True, random_state=42)`, no holdout |
| Django app | `run_ml_view.py:12,137-139`; `ml_models\processing.py:8-27` | `test_size=user (default 0.2), random_state=42` |

## Q6 — sentinel handling (prior art)

- S23 `Clean Combined Crash Data.ipynb` `clean_aadt`: `if float(entry) < 0: append(None)` (neutralises int32-min AADT). Folds `'Null value'`/`'Missing'` into per-field "Unknown" buckets; age `'More than 100 Years'→100`, clip [15,65]; Time of Day text→int hour.
- F24 `akCrashData-main\crashdata\dataModels\createDataModel.py:11,27-28`: regex `^(Null value|no Data|Undetermined|Not Specified|Not Reported|Missing|Unknown)$` + `fillna('Unknown')`.
- 2025 `Alaska-Crash-Analysis-main\...\data_cleaning\Script to Clean.py:10-40`: `UNKNOWN_STRINGS` set; profiles %unknown only, writes no cleaned file.
- **Milepoint:** no dedicated cleaning found (appears only in a `df.info()` dump, `Combine Crash Data.ipynb:237`).

## Caveats
1. Exact producers of both cleaned CSVs unrecovered (Colab/Drive); Year-drop is inference.
2. Author identities: an S23 contributor (Spring 2023, handle redacted); `rnmercado` -- the
   applicant (2025). Raw 09-12 xlsx predates both.
3. `poster_viz\data_queries.py` uses a Django ORM model absent from the F24 app → 2025-era, different schema.

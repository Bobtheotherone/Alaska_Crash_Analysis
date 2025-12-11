# AGENTS.md – Data-Prep & Cleaning Agent Guide

**Goal:**  
Keep the **“2. Data Preparation”** UI, the **`/api/models/run/`** contract, and the **backend cleaning pipeline** in sync.

After following this guide, an agent (or human) should be able to:

* Understand how the data-prep knobs flow **UI → API → worker → cleaning code → model**.
* Wire up currently-supported backend knobs that are **not yet UI-visible**.
* Fix knobs that are **UI-visible but not backend-connected**.
* Avoid breaking the existing ingestion gateway or model API.

---

## 1. End-to-end architecture (quick map)

High-level flow:

1. **Upload & quick validation (frontend)**  
   * React app in `alaska_ui` parses CSV/XLSX on the client and runs Peyton-style validation using `runValidationLogic` and `runClientValidation`. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}  
   * UI writes results into `ValidationResults`, which power the **Data Tables** view and the **column plan**. :contentReference[oaicite:2]{index=2}

2. **Upload to backend**  
   * `App.tsx` posts the file to `POST /api/upload/` (`analysis.views.upload_and_analyze`). :contentReference[oaicite:3]{index=3}  
   * Backend parses the file, saves an `ingestion.UploadedDataset`, and returns a summary plus `upload_id` (a.k.a. **dataset ID**). :contentReference[oaicite:4]{index=4}

3. **Start a model job**  
   * React calls `startModelJob(datasetId)` → `POST /api/models/run/` (`analysis.views.model_run`). :contentReference[oaicite:5]{index=5}  
   * The JSON body looks like:

     ```json
     {
       "dataset_id": "<uuid>",
       "model_name": "crash_severity_risk_v1",
       "parameters": {
         "cleaning": { ... },
         "model_params": { ... }
       }
     }
     ```

     `parameters` is stored verbatim in `ModelJob.parameters`. :contentReference[oaicite:6]{index=6}

4. **Worker**  
   * `enqueue_model_job(job.id)` (imported in `analysis.views`) runs `run_model_job` in `analysis/ml_core/worker.py`. :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}  
   * Worker:
     * Reloads the job and its `UploadedDataset`. :contentReference[oaicite:9]{index=9}  
     * Loads the raw file into a DataFrame via `load_dataframe_from_bytes`. :contentReference[oaicite:10]{index=10}  
     * Splits `parameters` into `cleaning_params` and `model_params`. :contentReference[oaicite:11]{index=11}  
     * Calls the appropriate trainer (`spec.trainer`) from `analysis/ml_core/models.py`. :contentReference[oaicite:12]{index=12}  
     * Stashes metrics, feature importances and **cleaning metadata** into `ModelJob.result_metadata`. :contentReference[oaicite:13]{index=13}

5. **Training & cleaning**  
   * Each trainer (`train_crash_severity_decision_tree`, `train_mrf`, etc.) calls `_ensure_cleaning_params(cleaning_params)` to normalize the `parameters.cleaning` object into kwargs for `build_ml_ready_dataset`. :contentReference[oaicite:14]{index=14}  
   * `build_ml_ready_dataset` in `analysis/ml_core/cleaning.py` performs:
     * Unknown value discovery.
     * Script-to-Clean column pruning (`unknown_threshold`, `yes_no_threshold`, `columns_to_drop`). :contentReference[oaicite:15]{index=15} :contentReference[oaicite:16]{index=16}  
     * Severity column selection (`guess_severity_column`) and mapping. :contentReference[oaicite:17]{index=17} :contentReference[oaicite:18]{index=18}  
     * Leakage detection (`find_leakage_columns_noninteractive`) + optional manual `leakage_columns`. :contentReference[oaicite:19]{index=19}  

---

## 2. Current Data-Prep UI state

### 2.1 DataPrepState

`alaska_ui/src/App.tsx` defines the `DataPrepState` interface used by the Data Prep card and validators: :contentReference[oaicite:20]{index=20}  

```ts
export interface DataPrepState {
  unknownThreshold: number;   // %
  yesNoThreshold: number;     // %
  speedLimit: number;         // MPH
  roadSurface: {
    dry: boolean;
    wet: boolean;
    iceSnow: boolean;
  };

  /**
   * User-chosen leakage columns (Peyton-style interactive flow).
   * These names are passed to the backend cleaning step and also
   * used to mark columns as Drop in the UI.
   */
  leakageColumnsToDrop: string[];
}
````

Default values: 

```ts
const [dataPrep, setDataPrep] = useState<DataPrepState>({
  unknownThreshold: 10,
  yesNoThreshold: 1,
  speedLimit: 70,
  roadSurface: { dry: true, wet: true, iceSnow: true },
  leakageColumnsToDrop: [],
});
```

### 2.2 “2. Data Preparation” card

`WorkflowPanel` renders the card shown in your screenshot: 

* Slider – **Unknown threshold** (`unknownThreshold`).
* Slider – **Yes/No imbalance threshold** (`yesNoThreshold`).
* Slider – **Max posted speed limit (MPH)** (`speedLimit`).
* Checkboxes – **Road surface** (`roadSurface.*`).
* Button – **Run Data Preparation** → `onDataPrepRun`. 

The card is disabled until Step 1 is complete (`isImportComplete`). 

### 2.3 How those knobs are actually used

1. **Unknown & Yes/No thresholds – used *only* on the client**

   * Used by `runValidationLogic` in `alaska_ui/src/lib/validator.ts` to compute `ColumnStat.status` (`Keep`/`Drop`) and `reason` based on Peyton’s rules.  
   * Also used by `runClientValidation` in `App.tsx` (richer version of same logic). 

   **They are *not* sent to the backend; the training pipeline always uses default thresholds from `ml_partner_adapters.config_bridge.UNKNOWN_THRESHOLD` and `YES_NO_THRESHOLD`.** 

2. **Leakage columns – used on client and server**

   * Selected via the interactive `window.confirm` + `window.prompt` flow executed inside `handleRunValidation`. 
   * Passed to `runValidationLogic` and `runClientValidation` via `applyLeakageOverrides`, which forces their status to `Drop` and appends an explanation.  
   * Sent to backend as `parameters.cleaning.leakage_columns` in `startModelJob`. 
   * In the backend, `_ensure_cleaning_params` passes them through to `build_ml_ready_dataset(leakage_columns=...)`.  

3. **Speed limit and road surface – currently NO-OP**

   * Only stored in `DataPrepState` and updated by UI handlers.  
   * Nowhere else in the frontend or backend reads `dataPrep.speedLimit` or `dataPrep.roadSurface.*`. There is no cleaning or filtering logic tied to these values.

---

## 3. Backend cleaning knobs that exist today

All cleaning is centralized in `analysis/ml_core/cleaning.py` and the model helpers in `analysis/ml_core/models.py`.

### 3.1 `build_ml_ready_dataset` parameters

Signature: 

```py
def build_ml_ready_dataset(
    df: pd.DataFrame,
    *,
    severity_col: str | None = None,
    base_unknowns: Iterable[str] | None = None,
    unknown_threshold: float | None = None,
    yes_no_threshold: float | None = None,
    leakage_columns: Iterable[str] | None = None,
    columns_to_drop: Iterable[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
```

Relevant behaviour:

* `severity_col`

  * If `None`, uses `guess_severity_column(df)` (prefers column literally named `"severity"`, otherwise first column containing `"severity"`). 
  * Protected from being dropped by Script-to-Clean.

* `base_unknowns`

  * If `None`, defaults to canonical `DEFAULT_UNKNOWN_STRINGS` built from `ml_partner_adapters`.  
  * Passed into `discover_unknown_placeholders`, which augments them based on data.  

* `unknown_threshold`, `yes_no_threshold`

  * If `None`, default to Peyton’s global `UNKNOWN_THRESHOLD` and `YES_NO_THRESHOLD` (0–100). 
  * Used in `clean_crash_dataframe_for_import` and `suggest_columns_to_drop` to drop high-unknown and highly imbalanced Yes/No columns.  

* `columns_to_drop`

  * Manually specified drop list; merged with automatically suggested drops. 

* `leakage_columns`

  * Manual leakage list; any features in this list are dropped *after* cleaning, in addition to automatically detected leakage columns. 

### 3.2 How HTTP `parameters.cleaning` is interpreted

`analysis/ml_core/models._ensure_cleaning_params` is the gatekeeper: 

```py
def _ensure_cleaning_params(cleaning_params: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    params = dict(cleaning_params or {})

    return {
        "severity_col": params.get("severity_col"),
        "base_unknowns": params.get("base_unknowns"),
        "unknown_threshold": params.get("unknown_threshold"),
        "yes_no_threshold": params.get("yes_no_threshold"),
        "columns_to_drop": params.get("columns_to_drop"),
        # Peyton-style manual leakage list; passed through to build_ml_ready_dataset
        "leakage_columns": params.get("leakage_columns"),
    }
```

And each trainer uses:

````py
cleaning_kwargs = _ensure_cleaning_params(cleaning_params)
X, y, cleaning_meta = build_ml_ready_dataset(df, **cleaning_kwargs)
``` :contentReference[oaicite:49]{index=49}  

**Key point:** The backend already supports `severity_col`, `base_unknowns`, `unknown_threshold`, `yes_no_threshold`, `columns_to_drop`, and `leakage_columns`. Only `leakage_columns` is currently wired from the UI.

---

## 4. Design goals for this agent

When modifying this area of the codebase, treat these as **non-negotiable invariants**:

1. **Single source of truth for cleaning thresholds**  
   * The sliders in the Data Prep card must be the same values used in **Script-to-Clean** on the backend.

2. **Frontend and backend agree on what’s dropped and why**  
   * Column plan / Data Tables should roughly match the columns that the worker actually drops. Minor differences (e.g. extra unknown tokens) are acceptable but should be intentional.

3. **Backwards compatibility**  
   * `POST /api/models/run/` must remain compatible with existing docs (`docs/model_api.md`). Adding optional fields is OK; breaking or renaming is not. :contentReference[oaicite:50]{index=50}  

4. **Data Prep card only contains meaningful knobs**  
   * Any knob in the “2. Data Preparation” section should influence either the cleaning pipeline or clearly labelled downstream filtering.

---

## 5. Concrete implementation plan

### 5.1 Extend DataPrepState with backend-visible knobs

Add the following fields to `DataPrepState` in `alaska_ui/src/App.tsx`: :contentReference[oaicite:51]{index=51}  

```ts
export interface DataPrepState {
  unknownThreshold: number;
  yesNoThreshold: number;
  speedLimit: number;
  roadSurface: {
    dry: boolean;
    wet: boolean;
    iceSnow: boolean;
  };

  leakageColumnsToDrop: string[];

  // NEW: manual non-leakage drops
  columnsToDrop: string[];

  // NEW: allow overriding which column is treated as severity
  severityColumn: string | null;

  // NEW: per-run extra "unknown" tokens (strings)
  additionalUnknownTokens: string[];
}
````

Update the default state accordingly:

```ts
const [dataPrep, setDataPrep] = useState<DataPrepState>({
  unknownThreshold: 10,
  yesNoThreshold: 1,
  speedLimit: 70,
  roadSurface: { dry: true, wet: true, iceSnow: true },
  leakageColumnsToDrop: [],
  columnsToDrop: [],
  severityColumn: null,
  additionalUnknownTokens: [],
});
```

Any place that resets `dataPrep` (e.g. `handleFileSelect`) must now also clear `columnsToDrop`, `severityColumn`, and `additionalUnknownTokens` while preserving other values. 

### 5.2 Wire thresholds + cleaning params to backend (`startModelJob`)

Modify `startModelJob` in `App.tsx` to send a richer `parameters.cleaning` payload. Current implementation only sends `leakage_columns`. 

Replace the body construction with something like:

```ts
const startModelJob = useCallback(
  async (datasetId: string) => {
    console.log("[UI] Starting model job for dataset:", datasetId);

    // Build cleaning payload expected by analysis.ml_core.models._ensure_cleaning_params
    const cleaning: any = {
      leakage_columns: dataPrep.leakageColumnsToDrop,
      unknown_threshold: dataPrep.unknownThreshold,
      yes_no_threshold: dataPrep.yesNoThreshold,
    };

    if (dataPrep.columnsToDrop.length > 0) {
      cleaning.columns_to_drop = dataPrep.columnsToDrop;
    }

    if (dataPrep.severityColumn) {
      cleaning.severity_col = dataPrep.severityColumn;
    }

    if (dataPrep.additionalUnknownTokens.length > 0) {
      // Merge UI-known base tokens with user-specified extras.
      const base = new Set<string>();
      BASE_UNKNOWN_TOKENS.forEach((tok) => base.add(tok));
      dataPrep.additionalUnknownTokens.forEach((tok) => {
        const norm = tok.trim().toLowerCase();
        if (norm) base.add(norm);
      });
      cleaning.base_unknowns = Array.from(base);
    }

    const body = {
      dataset_id: datasetId,
      model_name: selectedModelName,
      parameters: {
        cleaning,
        model_params: {},
      },
    };

    const resp = await fetch("/api/models/run/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...buildAuthHeader(),
      },
      body: JSON.stringify(body),
    });
    ...
  },
  [auth.username, auth.password, selectedModelName, dataPrep]
);
```

> Why this works: `_ensure_cleaning_params` pulls exactly these keys and forwards them to `build_ml_ready_dataset`.  That function, in turn, passes `unknown_threshold`, `yes_no_threshold`, and `columns_to_drop` into `clean_crash_dataframe_for_import`, and treats `severity_col`, `base_unknowns`, and `leakage_columns` as documented above.   

No backend code changes are required for these fields; only the UI must start sending them.

### 5.3 Add severity column override UI

Use the existing `ValidationResults` to populate a dropdown listing column names.

In `WorkflowPanel.tsx`:

1. Compute available columns:

   ```ts
   const availableColumns =
     validationResults?.columnStats.map((cs) => cs.column) ?? [];
   ```

   Place this near the top of the component body, after `getStatus` / before the `return`. 

2. Inside the Data Prep `<fieldset>`, add a new block after the road-surface controls:

   ```tsx
   {availableColumns.length > 0 && (
     <div>
       <label
         htmlFor="severity-column"
         className="block font-medium text-gray-700 mb-1"
       >
         Severity / outcome column
       </label>
       <select
         id="severity-column"
         className="w-full bg-white border border-neutral-medium rounded px-2 py-1 text-sm"
         value={dataPrepState.severityColumn ?? ""}
         onChange={(e) =>
           onDataPrepChange({
             severityColumn: e.target.value || null,
           })
         }
       >
         <option value="">
           Auto-detect (prefer column named &quot;severity&quot;)
         </option>
         {availableColumns.map((col) => (
           <option key={col} value={col}>
             {col}
           </option>
         ))}
       </select>
       <p className="mt-1 text-xs text-gray-500">
         If left blank, the backend will guess the severity column by name.
       </p>
     </div>
   )}
   ```

3. Ensure `DataPrepState` includes `severityColumn` and that you’ve added it to the default state as described in §5.1.

When this dropdown is non-empty, the value will be shipped as `parameters.cleaning.severity_col` and fed into `build_ml_ready_dataset`, which passes it through to `clean_crash_dataframe_for_import` and protects that column from being dropped.  

### 5.4 Add manual `columns_to_drop` UI

Goal: Let users drop additional low-value columns even if they don’t exceed the unknown/imbalance thresholds.

**Minimal implementation (text field):**

In the Data Prep card, still inside the `<fieldset>`, add a simple comma-separated text box:

```tsx
<div>
  <label
    htmlFor="manual-drop-columns"
    className="block font-medium text-gray-700"
  >
    Additional columns to drop
  </label>
  <p className="text-xs text-gray-500 mb-1">
    Comma-separated list of columns to always drop before modeling.
  </p>
  <input
    id="manual-drop-columns"
    type="text"
    className="w-full border border-neutral-medium rounded px-2 py-1 text-sm"
    placeholder="Example: VIN, CrashReportId"
    value={dataPrepState.columnsToDrop.join(", ")}
    onChange={(e) =>
      onDataPrepChange({
        columnsToDrop: e.target.value
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean),
      })
    }
  />
</div>
```

This populates `dataPrep.columnsToDrop`, which is then sent as `parameters.cleaning.columns_to_drop` (see §5.2). Backend logic in `clean_crash_dataframe_for_import` already unions these with automatically suggested drops: 

```py
auto_drop = suggest_columns_to_drop(...)
user_specified_drops: Set[str] = set(columns_to_drop or [])
drop_cols = auto_drop | user_specified_drops
```

**Optional UX alignment:**
To have the **Data Tables** view reflect these manual drops, you can generalize `applyLeakageOverrides` into `applyColumnOverrides(results, { leakageColumns, manualDropColumns })` that:

* Forces both leakage columns and `columnsToDrop` to `status = "Drop"`.
* Appends a human-readable reason for manual drops (e.g., “User-marked as low-value (manual drop).”).

This requires updating:

* The helper in `App.tsx`. 
* The call sites in `handleRunValidation` where `applyLeakageOverrides` is used. 

### 5.5 Add “extra unknown tokens” UI and align with backend

**Frontend control (Data Prep card):**

Add:

```tsx
<div>
  <label
    htmlFor="extra-unknowns"
    className="block font-medium text-gray-700"
  >
    Additional values to treat as &quot;unknown&quot;
  </label>
  <p className="text-xs text-gray-500 mb-1">
    Comma-separated; e.g. &quot;UNK, 99, 9999&quot;.
  </p>
  <input
    id="extra-unknowns"
    type="text"
    className="w-full border border-neutral-medium rounded px-2 py-1 text-sm"
    value={dataPrepState.additionalUnknownTokens.join(", ")}
    onChange={(e) =>
      onDataPrepChange({
        additionalUnknownTokens: e.target.value
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean),
      })
    }
  />
</div>
```

**Client-side validators:**

Update `alaska_ui/src/lib/validator.ts` so that the discovery function merges in `config.additionalUnknownTokens`.

1. Change the signature of `discoverUnknownPlaceholders`:

   ```ts
   const discoverUnknownPlaceholders = (
     data: Record<string, string>[],
     extraUnknowns: string[] = []
   ): Set<string> => {
     ...
     const frequentNewUnknowns = new Set<string>();
     ...
     const extras = extraUnknowns
       .map((s) => s.trim().toLowerCase())
       .filter((s) => !!s);

     return new Set([
       ...BASE_UNKNOWN_STRINGS,
       ...frequentNewUnknowns,
       ...extras,
     ]);
   };
   ```

2. Pass `config.additionalUnknownTokens` from `runValidationLogic`: 

   ```ts
   const augmentedUnknowns = discoverUnknownPlaceholders(
     data,
     config.additionalUnknownTokens || []
   );
   ```

3. Optionally, do similar wiring for `discoverUnknownTokens` in `App.tsx`’s `runClientValidation` path, or refactor to share the same helper. 

**Backend alignment:**

As in §5.2, you’re already sending:

```ts
cleaning.base_unknowns = Array.from(base); // BASE_UNKNOWN_TOKENS + additionalUnknownTokens
```

`build_ml_ready_dataset` will pass `base_unknowns` through to `clean_crash_dataframe_for_import`, which then calls `discover_unknown_placeholders` with the same seed list. 

This keeps “unknown” semantics largely aligned between frontend and backend.

### 5.6 Expose leakage list in the Data Prep card (read-only)

Even if editing stays in the Step-1 prompt, it’s helpful for users to *see* which columns will be treated as leakage during modeling.

In `WorkflowPanel.tsx`, under the other Data Prep controls, add:

```tsx
{dataPrepState.leakageColumnsToDrop.length > 0 ? (
  <div className="text-xs text-gray-600">
    <span className="font-medium">Leakage columns:</span>{" "}
    {dataPrepState.leakageColumnsToDrop.join(", ")}
  </div>
) : (
  <p className="text-xs text-gray-500">
    No leakage columns selected yet. You can mark them during validation.
  </p>
)}
```

The actual selection flow remains in `handleRunValidation` (interactive prompt). 

If you later want editing from Step 2, refactor the leakage-prompt logic into a reusable function that can be called both from the validation step and from an “Edit leakage columns…” button.

### 5.7 Handling `speedLimit` and `roadSurface` knobs

These controls are **currently not connected** to any backend behaviour:

* No cleaning parameter uses them.
* They are not used when loading `CrashRecord` or building `X, y`.  

To **resolve the mismatch** while keeping the code maintainable:

1. **Short-term (recommended default):**

   * Document in the UI (small text under the controls) that they are **map-filter preferences only** and do not affect model training yet.
   * Optionally, hide or disable them in the Data Prep card until a map-filter pipeline is implemented.

2. **Future work (if you choose to wire them):**

   * Introduce a filter layer in the worker or model code that subsets the DataFrame before calling `build_ml_ready_dataset`, using:

     * `posted_speed_limit` for `speedLimit` (see `CrashRecord.posted_speed_limit`). 
     * A normalized “road surface” column for `roadSurface`.
   * Pass the chosen values via `parameters.model_params` (not `cleaning`), since they are **row filters**, not column-cleaning rules.

---

## 6. Testing & verification checklist

When implementing or modifying these behaviours, validate with the following steps:

1. **Unit tests (Python)**

   * Add tests under `analysis/tests/` that:

     * Call `_ensure_cleaning_params` with a dict containing all keys and assert the result matches expectations. 
     * Call `build_ml_ready_dataset` with explicit `unknown_threshold` / `yes_no_threshold` and verify that columns above/below thresholds are dropped as expected.
     * Verify that `severity_col` override is respected and protected from dropping, and that an invalid `severity_col` raises a clear error. 

2. **Integration test (Python)**

   * Create a `ModelJob` with `parameters={"cleaning": {...}}` including:

     * `unknown_threshold`
     * `yes_no_threshold`
     * `columns_to_drop`
     * `leakage_columns`
     * `severity_col`
   * Run `run_model_job(job.id)` and assert that:

     * `job.result_metadata["cleaning_meta"]["cleaning_config"]` reflects your thresholds. 
     * `job.result_metadata["leakage_warnings"]["leakage_columns"]` matches the leakage list. 

3. **Frontend manual tests**

   * Start dev env with `start_app.bat` (or equivalent). 
   * Upload a small CSV and run validation.
   * Change the Data Prep sliders:

     * Confirm that column statuses in the **Data Tables** view update when re-running Step 2.
   * Run a model:

     * Inspect the `POST /api/models/run/` request in the browser dev tools:

       * `parameters.cleaning.unknown_threshold` matches the slider.
       * `yes_no_threshold`, `columns_to_drop`, `severity_col`, `base_unknowns`, and `leakage_columns` are present when set.
   * Poll `GET /api/models/results/<job_id>/` and confirm that:

     * The job succeeds.
     * `result_metadata.cleaning_meta.cleaning_config` matches the UI values. 

---

## 7. Non-goals and guardrails

* **Do not** change the shape of the `ModelJob.parameters` JSON beyond adding optional keys in `parameters.cleaning` and `parameters.model_params`. The docs in `docs/model_api.md` must remain valid. 
* **Do not** move cleaning logic out of `analysis.ml_core.cleaning`; keep a single shared cleaning stack for ETL and modeling. 
* **Do not** make frontend-only “fixes” (e.g., changing thresholds only in React) without also ensuring the backend honours the same values when training models.

---

If you’re an agent working in this repo, treat this document as the **source of truth** for data-prep / cleaning behaviour. When in doubt, keep the three layers aligned:

* Data Prep UI (`alaska_ui`),
* Model API (`/api/models/run/`), and
* Cleaning stack (`analysis/ml_core`).

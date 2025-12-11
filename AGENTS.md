# AGENTS.md – Map & Geospatial Integration Guide

Audience: **AI coding assistants** (Codex, Copilot, ChatGPT, etc.) and future developers working in this repo.

Scope: how to correctly use the existing geospatial backend (PostGIS + Django) and wire it to a fully-featured, interactive **Map** tab in the React UI. This document is the canonical reference for anything involving maps, crash locations, or geospatial filters.

---

## 1. System overview (what already exists)

### 1.1 Backend

* Project: Django with GeoDjango + PostGIS.
* App: `crashdata`
  * Model `CrashRecord` includes:
    * `location`: `PointField` (SRID=4326, lon/lat), indexed with GiST.
    * `severity`: MMUCC KABCO codes (`"K"|"A"|"B"|"C"|"O"`).
    * `crash_datetime`, `roadway_name`, `municipality`, `posted_speed_limit`, counts, etc.
  * ETL command `import_crash_records <upload_id>` maps cleaned CSV → `CrashRecord`, setting `location` from `longitude`/`latitude` (or `None` if invalid).

Geospatial endpoints (defined in `crashdata/urls.py`):

```python
urlpatterns = [
    path("severity-histogram/", views.severity_histogram_view, name="crashdata-severity-histogram"),
    path("crashes-within-bbox/", views.crashes_within_bbox_view, name="crashdata-crashes-within-bbox"),
    path("heatmap/", views.heatmap_view, name="crashdata-heatmap"),
    path("exports/crashes.csv", views.export_crashes_csv, name="crashdata-export-crashes-csv"),
]
````

These are served under `/api/crashdata/…` by the project router.

> **Important:** The README still references older paths like `/api/crashdata/within-bbox/` and `/api/crashdata/export/`. Treat **`crashdata/urls.py` as ground truth**. If you change endpoints, update both the code and the README in a single PR.

### 1.2 Frontend

* Vite + React app in `alaska_ui`.

* Main layout component: `alaska_ui/src/components/MainContent.tsx`.

  * Tabs: `['Map', 'Data Tables', 'Report Charts', 'Classifications', 'EBM']`.
  * **Map tab currently renders only a static OpenStreetMap `<iframe>`** and does not call any backend geospatial APIs.

* Root app component: `alaska_ui/src/App.tsx`.

  * Holds canonical state:

    * `uploadId: string | null` – backend ID of the selected/validated upload.
    * `validationResults: ValidationResults | null` – includes ingestion gateway summary and row checks.
    * `analysisResults`, `dataPrep`, `selectedModelName`, etc.

  * `ValidationResults` type includes:

    ```ts
    export interface ValidationResults {
      // client-side stats...
      ingestionOverallStatus?: 'accepted' | 'rejected' | 'pending' | string;
      ingestionMessage?: string;
      ingestionErrorCode?: string;
      ingestionRowChecks?: {
        total_rows: number;
        invalid_row_count: number;
        invalid_geo_row_count: number;
      };
      ingestionSteps?: {
        step: string;
        message: string;
        status: 'passed' | 'failed' | 'skipped';
        severity?: 'info' | 'warning' | 'error';
        code?: string;
        is_hard_fail?: boolean;
      }[];
    }
    ```

  * `extractIngestionSummary(...)` normalizes the backend’s ingestion/geo row counts into `validationResults.ingestionRowChecks`.

**Key point:** backend + ETL already have rich geospatial support. The UI just isn’t using it yet.

---

## 2. Backend geospatial API contracts

When wiring up or extending map behavior, **do not change these contracts** unless you also:

1. Update `crashdata/urls.py` and `crashdata/views.py`.
2. Update the README / docs.
3. Update any frontend callers.

### 2.1 `GET /api/crashdata/crashes-within-bbox/`

Purpose: return crashes as a GeoJSON `FeatureCollection` of points.

**Required query params**

* `min_lon`, `min_lat`, `max_lon`, `max_lat` – floats (WGS84 degrees).

Return `400` if any are missing or not floats.

**Optional query params**

* `upload_id` – UUID of `ingestion.UploadedDataset`.
* `severity` – comma-separated list of KABCO letters, e.g. `"K,A,B"`.
* `municipality` – string; filters `CrashRecord.municipality__iexact`.
* `start_datetime`, `end_datetime` – ISO-8601 datetime **or** date; parsed by `_parse_datetime_param`.

  * Dates are treated as midnight UTC.
  * If either parameter cannot be parsed, returns `400`.
* `limit` – integer; default 5,000; clamped to `1–50,000`. If outside, resets to 5,000.

**Response shape (success)**

```json
{
  "type": "FeatureCollection",
  "count": <number of features returned>,
  "limit": <limit used>,
  "bbox": [min_lon, min_lat, max_lon, max_lat],
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [lon, lat]
      },
      "properties": {
        "id": <CrashRecord.id>,
        "crash_id": "<string>",
        "severity": "K|A|B|C|O",
        "crash_datetime": "<ISO datetime>",
        "roadway_name": "<string>",
        "municipality": "<string>",
        "posted_speed_limit": <int | null>,
        "dataset_id": "<upload UUID as string>"
      }
    },
    ...
  ]
}
```

Notes:

* Rows with `location=None` are **silently skipped**; they never appear as features.
* Features come from `queries.crashes_within_bbox(...)` with exactly the filters listed above.

### 2.2 `GET /api/crashdata/heatmap/`

Purpose: return a lightweight crash density grid for the current bounding box.

**Required query params**

* Same bbox as above: `min_lon`, `min_lat`, `max_lon`, `max_lat`.

**Optional query params**

* `upload_id`, `severity`, `municipality`, `start_datetime`, `end_datetime` (same semantics as `/crashes-within-bbox/`).
* `grid_size` – float in degrees; default `0.05`. If `<= 0`, reset to `0.05`.

Backend behavior:

* Re-uses `queries.crashes_within_bbox(...)` but only selects `location`.
* Buckets each crash into integer grid cells via `x = int(lon // grid_size)`, `y = int(lat // grid_size)`.
* Returns per-cell counts.

**Response shape**

```json
{
  "bbox": [min_lon, min_lat, max_lon, max_lat],
  "grid_size": 0.05,
  "cells": [
    {
      "count": 12,
      "center": [lon, lat],
      "grid_size": 0.05
    },
    ...
  ]
}
```

The UI is expected to render these as either:

* discrete colored squares (approximate cell extents from `center` and `grid_size`), or
* inputs to a heatmap layer.

### 2.3 `GET /api/crashdata/exports/crashes.csv`

Purpose: CSV export for offline analysis / external GIS, honoring the same conceptual filters.

**Required query params**

* `upload_id` – UUID.

**Optional query params**

* `severity`, `municipality`, `start_datetime`, `end_datetime` as above.
* `max_rows` – integer; default 100,000; clamped to `1–500,000`.

Behavior:

* Filters `CrashRecord` by dataset + optional filters.

* Counts rows; if `row_count > max_rows`, returns `400` with a JSON body explaining that max_rows was exceeded.

* Otherwise streams CSV with the following header:

  ```csv
  id,dataset_id,crash_id,crash_datetime,severity,roadway_name,municipality,posted_speed_limit,vehicle_count,person_count,lon,lat
  ```

* Cells that could be interpreted as formulas are prefixed with `'` in the CSV to avoid Excel CSV injection (`_safe_csv_value`).

### 2.4 Authentication

All crashdata endpoints are decorated with:

```python
@api_view(["GET"])
@permission_classes([IsAuthenticated])
```

The frontend currently uses **Basic Auth** headers built in `App.tsx`:

```ts
const buildAuthHeader = () => {
  if (!auth.username || !auth.password) return {};
  const encoded = btoa(`${auth.username}:${auth.password}`);
  return { Authorization: `Basic ${encoded}` };
};
```

Any new fetches from the map UI must **reuse this mechanism**.

---

## 3. Desired Map UI behavior

The current Map tab is just an embedded OSM basemap. The goal is:

> **A fully interactive map of Alaska that shows crash points and crash density for the currently selected upload, shares filters with the rest of the app, surfaces geo quality issues, and can export map-filtered data.**

### 3.1 Functional requirements

When `uploadId` is set (i.e., ingestion + ETL have completed):

1. **Basemap**

   * Use a real JS map library (e.g. `react-leaflet`, MapLibre, Mapbox GL) instead of an `<iframe>`.
   * Default center/zoom should show Alaska; reuse the existing OSM bbox as a starting view.

2. **Crash points layer (`/crashes-within-bbox/`)**

   * On initial render and on each pan/zoom, call:

     ```
     GET /api/crashdata/crashes-within-bbox/
       ?upload_id=<uploadId>
       &min_lon=<west>&min_lat=<south>
       &max_lon=<east>&max_lat=<north>
       [&severity=K,A,B]
       [&municipality=Anchorage]
       [&start_datetime=2024-01-01]
       [&end_datetime=2024-12-31]
       [&limit=5000]   // optional
     ```

   * Render each feature as a marker.

     * Color markers by `properties.severity` (e.g., K/A red/orange, B/C yellow, O gray).
     * Popups should show at minimum: `crash_id`, `crash_datetime`, `roadway_name`, `municipality`, `posted_speed_limit`.

   * If `count === limit`, show a small UI notice that the view may be truncated and the user should zoom or filter further.

3. **Heatmap layer (`/heatmap/`)**

   * Add a layer toggle UI (checkboxes or buttons):

     * `[x] Crash points`
     * `[ ] Crash density`

   * When the **density** layer is enabled, call:

     ```
     GET /api/crashdata/heatmap/
       ?upload_id=<uploadId>
       &min_lon=<west>&min_lat=<south>
       &max_lon=<east>&max_lat=<north>
       &grid_size=<derivedFromZoom>    // e.g., 0.25 at zoomed out, 0.05 at mid, 0.01 zoomed in
       [&severity=...]
       [&municipality=...]
       [&start_datetime=...]
       [&end_datetime=...]
     ```

   * Render each cell either as:

     * a square polygon approximated from `center` and `grid_size`, colored on a continuous scale by `count`, or
     * a point in an actual heatmap layer with weight = `count`.

4. **Filters**

   * Map filters must be kept in React state and **reused across endpoints**:

     * severity (multi-select)
     * municipality (text input or dropdown)
     * date range (start/end date or datetime)
   * These should eventually stay in sync with the rest of the app (charts, classifications). Until a global filter system exists, it’s acceptable to keep them map-local but designed in a way that they can easily be lifted to top-level state.

5. **Geo-quality surfacing**

   * Above the map (or in the left workflow panel) show a short summary derived from `validationResults.ingestionRowChecks`:

     * total rows
     * invalid rows (generic data issues)
     * invalid_geo_row_count (rows outside Alaska bounds or with invalid coordinates)

   * Example text:

     > **Geo coverage:** 12,345 total rows, 123 with invalid values, 42 with invalid / out-of-bounds coordinates. Only crashes with valid coordinates are shown on the map.

   * If `invalid_geo_row_count > 0`, also show a non-blocking warning banner or icon in the Map tab.

6. **No dataset selected**

   * If `uploadId === null` or no crash records have been imported yet, show an empty-state message in the Map tab:

     > “Import and validate a dataset, then run the ETL to see crashes on the map.”

   * You may reuse an existing pattern from other tabs for this “nothing to show yet” state.

7. **Export shortcut**

   * Provide a “Export map-filtered crashes (CSV)” button in the map UI that calls:

     ```
     GET /api/crashdata/exports/crashes.csv
       ?upload_id=<uploadId>
       [&severity=...]
       [&municipality=...]
       [&start_datetime=...]
       [&end_datetime=...]
     ```

   * This should initiate a file download in the browser.

   * If the backend responds with `400` and a JSON body complaining about `row_count > max_rows`, display that message to the user and suggest narrowing filters.

### 3.2 Non-functional requirements

* **Do not break other tabs.**

  * `MainContent` should continue to pass the same props to `Data Tables` / `Report Charts` / `Classifications`.
* **Reuse auth / base URL conventions.**

  * All map fetches must use `buildAuthHeader()` from `App.tsx`.
* **Error handling.**

  * For HTTP errors:

    * `401/403`: show “You may not be logged in or lack access to this dataset.”
    * `400`: show the `detail` message from the response if present.
  * For network failures: show a compact inline error and log the full error to `console.error`.

---

## 4. Frontend implementation blueprint

When asked to “hook the map up” or “add new map features”, follow this plan unless the user explicitly requests a different architecture.

### 4.1 New component: `CrashMap`

Create `alaska_ui/src/components/CrashMap.tsx` with roughly this public interface:

```ts
interface CrashMapProps {
  uploadId: string | null;
  authHeader: () => Record<string, string>; // wrapper around buildAuthHeader
  // optional: shared filters
  initialSeverity?: string[];         // ['K', 'A', 'B', ...]
  initialMunicipality?: string;
  initialStartDate?: string;          // ISO date or datetime
  initialEndDate?: string;
  ingestionRowChecks?: {
    total_rows: number;
    invalid_row_count: number;
    invalid_geo_row_count: number;
  } | undefined;
}
```

Responsibilities of `CrashMap`:

1. Manage **viewport state** (center, zoom, bbox) using your chosen map library.
2. Manage **local filter state** (severity, municipality, date range) until a global filter store exists.
3. On viewport or filter change, fetch:

   * current crash points from `/api/crashdata/crashes-within-bbox/`
   * current heatmap from `/api/crashdata/heatmap/` when density layer is toggled on
4. Render:

   * base map
   * markers and/or heatmap cells
   * status bar above the map showing ingestion geo summary and any fetch errors.

Map wiring inside `MainContent.tsx`:

Replace the current `Map` tab branch:

```tsx
case 'Map':
  return (
    <div className="h-full w-full bg-gray-200 rounded-lg overflow-hidden">
      <iframe ... />
    </div>
  );
```

with:

```tsx
case 'Map':
  return (
    <CrashMap
      uploadId={uploadId}
      authHeader={buildAuthHeader}
      ingestionRowChecks={validationResults?.ingestionRowChecks}
      // if/when you lift filters to App, pass them here too
    />
  );
```

You’ll need to plumb `uploadId`, `buildAuthHeader`, and `validationResults` through `MainContent`’s props from `App.tsx`.

### 4.2 Data fetching inside `CrashMap`

Use `fetch` with the auth header:

```ts
const headers = {
  ...authHeader(),
  'Accept': 'application/json',
};

const params = new URLSearchParams({
  upload_id: uploadId,
  min_lon: bbox.minLon.toString(),
  min_lat: bbox.minLat.toString(),
  max_lon: bbox.maxLon.toString(),
  max_lat: bbox.maxLat.toString(),
});

if (severity.length) params.set('severity', severity.join(','));
if (municipality) params.set('municipality', municipality);
if (startDate) params.set('start_datetime', startDate);
if (endDate) params.set('end_datetime', endDate);

const resp = await fetch(`/api/crashdata/crashes-within-bbox/?${params.toString()}`, {
  method: 'GET',
  headers,
});
```

Guidelines:

* Always guard against `uploadId === null` — do nothing and show an empty-state.
* Debounce fetches on viewport change (e.g., wait for the user to stop panning for ~200–300 ms).
* Cancel in-flight requests when new ones are issued (e.g., by tracking an `AbortController` per fetch).

### 4.3 Coordinate conventions

* The backend uses WGS84 longitude/latitude (SRID 4326).
* All bbox and coordinate arrays are `[lon, lat]`.
* When you convert between map library bounds and API params:

  * `min_lon` = west, `min_lat` = south
  * `max_lon` = east, `max_lat` = north

Double-check that the library’s `getBounds()` returns the same order; different map libraries sometimes use `(southWest, northEast)` or `(lat, lon)` tuples.

### 4.4 Using ingestion geo stats in the Map

Within `CrashMap`, if `ingestionRowChecks` is provided:

```ts
const { total_rows, invalid_row_count, invalid_geo_row_count } =
  ingestionRowChecks ?? { total_rows: 0, invalid_row_count: 0, invalid_geo_row_count: 0 };
```

Render a small summary above the map:

* Example:

  > **Data quality:** 12,345 rows (12 invalid, 42 invalid coordinates – excluded from map)

* Only show the warning highlight if `invalid_geo_row_count > 0`.

Do **not** attempt to “fix” or drop rows yourself on the frontend; trust the backend’s ETL + validation logic.

---

## 5. Testing and validation

When you implement or modify map behavior, ensure at least:

1. **Smoke tests / manual checks**

   * Upload a sample dataset, run ingestion + ETL, log into the UI, and:

     * Verify that the Map tab shows crash markers in the expected locations.
     * Zoom in/out and pan; markers and/or heatmap must update accordingly.
     * Change severity, municipality, and date filters and confirm server calls include the proper query parameters.
     * Trigger a `limit` hit by zooming way out on a dense dataset and confirm the explanatory UI message appears.

2. **Error paths**

   * Intentionally break the auth header (wrong password) and confirm 401/403 are handled gracefully.
   * Temporarily force the backend to return `400` (e.g. by passing bad date strings) and confirm the UI surfaces `detail`.

3. **Consistency**

   * Confirm that `invalid_geo_row_count` in the ingestion panel matches the text shown in the map geo-quality summary.
   * Confirm that a CSV exported from the map with given filters matches the points shown (modulo `max_rows` caps).

If you add automated tests:

* For backend: unit tests for `crashes_within_bbox_view` and `heatmap_view` should cover:

  * bbox validation
  * date parsing
  * severity/municipality filters
  * `limit` and `grid_size` sanity.
* For frontend: test that `CrashMap` builds the correct URLs from a given bbox and filter state and that it renders a reasonable number of markers/heat cells for a mocked response.

---

## 6. Common pitfalls (and how to avoid them)

When working on this area, **do not**:

1. **Change endpoint paths** (`crashes-within-bbox/`, `heatmap/`, `exports/crashes.csv`) by hand in the frontend.

   * If a bug appears to be “wrong path,” confirm against `crashdata/urls.py` before “fixing” anything.
2. **Swap lat/lon order.**

   * API, DB, and GeoJSON are consistently `[lon, lat]`. If the map looks mirrored or offset, check your order.
3. **Ignore auth.**

   * All crashdata endpoints require `IsAuthenticated`. Always include the Basic Auth header from `App.tsx`.
4. **Duplicate state.**

   * Treat `uploadId` in `App.tsx` as the single source of truth. Do not add another upload ID in `CrashMap` or `MainContent`.
5. **Hide geo problems.**

   * Don’t silently ignore `invalid_geo_row_count`. Always surface it in the UI so analysts know some rows are off-map.

---

## 7. Extending beyond the basics

Once the core wiring is correct, additional features are welcome, but they MUST reuse the contracts above:

* Add clustering for point markers at low zooms if needed.
* Add clickable selection that cross-highlights a crash in both Map and Data Tables (requires a shared “selected crash” state).
* When models start returning spatially indexed results (e.g., risk scores per segment/grid cell), add new overlays **on top of** the existing crash + heatmap layers rather than replacing them.

---

## 8. Summary checklist for Codex / assistants

Before you propose code changes related to maps:

1. **Confirm** the endpoint URLs in `crashdata/urls.py`.
2. **Use** `uploadId` from `App.tsx` as the dataset key.
3. **Include** Basic Auth headers using `buildAuthHeader()`.
4. **Pass** map filters as query params exactly matching the backend (`severity`, `municipality`, `start_datetime`, `end_datetime`, `limit`, `grid_size`).
5. **Respect** `invalid_geo_row_count` and communicate geo coverage in the Map tab.
6. **Avoid** touching unrelated tabs or changing backend contracts unless the user explicitly asks for it and you update docs/tests accordingly.

If you follow this document, all the previously identified gaps (static map, no crash markers, no heatmap, no exposure of geo quality, no CSV export from map filters) should be fully resolved and the map UI will correctly leverage the existing geospatial backend.

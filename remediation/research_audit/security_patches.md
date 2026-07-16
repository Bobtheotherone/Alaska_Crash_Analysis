# Security & data-integrity patches (SEC-003, SW-001, SW-007, SW-008)

These are the concrete corrections for the research-relevant security/integrity defects.
They are **specified and reviewed against the actual source** but **not executed here**,
because verifying them requires a running Django + PostGIS test database (not available in
this environment without provisioning a DB). Each is small, local, and testable; the
acceptance test is stated so a reviewer can add it.

---

## SEC-003 — Owner-scope every crash query; require a dataset id

**Defect:** `crashdata/queries.py` base querysets are `CrashRecord.objects.all()` and the
`dataset` argument is optional; `crashdata/views.py` histogram/bbox/heatmap treat `upload_id`
as optional and skip the ownership check when it is omitted, so an authenticated user can
read aggregates/points across *all* users' datasets.

**Fix (queries.py):** make `dataset` **required** and never start from `.all()` unscoped.

```python
# crashdata/queries.py  — require a dataset (no global base queryset)
def severity_histogram(dataset: UploadedDataset, *, municipality=None,
                       start_datetime=None, end_datetime=None):
    if dataset is None:
        raise ValueError("severity_histogram requires a dataset (owner-scoped).")
    qs = CrashRecord.objects.filter(dataset=dataset)          # was: CrashRecord.objects.all()
    qs = _apply_common_filters(qs, municipality=municipality,
                               start_datetime=start_datetime, end_datetime=end_datetime)
    return qs.values("severity").annotate(count=Count("id")).order_by("severity")

def crashes_within_bbox(dataset: UploadedDataset, min_lon, min_lat, max_lon, max_lat, **kw):
    if dataset is None:
        raise ValueError("crashes_within_bbox requires a dataset (owner-scoped).")
    bbox = Polygon.from_bbox((min_lon, min_lat, max_lon, max_lat)); bbox.srid = 4326
    qs = (CrashRecord.objects.filter(dataset=dataset)          # scope FIRST
          .exclude(location__isnull=True)
          .annotate(loc_geom=Cast("location", output_field=gis_models.GeometryField(srid=4326)))
          .filter(loc_geom__within=bbox))
    return _apply_common_filters(qs, **kw)
```

**Fix (views.py):** make `upload_id` required on histogram/bbox/heatmap and always resolve
it through the ownership check:

```python
upload_id = request.query_params.get("upload_id")
if not upload_id:
    return Response({"detail": "upload_id is required."}, status=400)
dataset = _get_dataset_for_user(_parse_uuid(upload_id), request.user)  # 404 if not owner/admin
```

**Acceptance (two-user integration test):**

```python
def test_user_a_cannot_read_user_b_crashes(self):
    ds_b = make_dataset(owner=user_b); import_rows(ds_b, N=10)
    self.client.force_authenticate(user_a)
    # omitted id -> 400, not global data
    assert self.client.get("/api/crashdata/severity-histogram/").status_code == 400
    # explicit foreign id -> 404
    r = self.client.get(f"/api/crashdata/severity-histogram/?upload_id={ds_b.id}")
    assert r.status_code == 404
```

---

## SW-007 / SW-008 — Atomic import + logical uniqueness

**Defect:** `crashdata/importer.py` deletes existing rows then `bulk_create`s new ones with
no `transaction.atomic()` (a mid-import failure empties the dataset), and `CrashRecord` has
no uniqueness constraint on `(dataset, crash_id)`.

**Fix (importer.py):** validate first, then replace atomically.

```python
from django.db import transaction

def import_crash_records_for_dataset(dataset, ...):
    records = _build_records(dataset, ...)          # parse + validate BEFORE deleting
    with transaction.atomic():
        CrashRecord.objects.filter(dataset=dataset).delete()
        CrashRecord.objects.bulk_create(records, batch_size=1000)
    return summary
```

**Fix (models.py + migration):**

```python
class CrashRecord(models.Model):
    ...
    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["dataset", "crash_id"],
                                    name="uniq_dataset_crashid"),
        ]
        indexes = [...]
```

**Acceptance:** (a) forcing an exception inside the atomic block leaves the original rows
intact (rollback test); (b) importing the same file twice is idempotent (row count stable);
(c) inserting a duplicate `(dataset, crash_id)` raises `IntegrityError`.

---

## SW-001 — One validated ingestion boundary

**Defect:** `analysis/views.py::upload_and_analyze` persists an `UploadedDataset` after only
dataframe/schema validation, bypassing the extension/size/MIME/AV gates in
`ingestion/views.py::upload_dataset`.

**Fix (specified):** extract the ingestion gate sequence into a reusable service
`ingestion/gateway.py::validate_and_store(upload, owner, *, require_persist)` and call it
from *both* endpoints; or make quick-analysis non-persistent (analyse in memory, store
nothing) and bounded by the same size/type checks. Add an endpoint-gate matrix test asserting
every persisted/model upload passed identical hard gates.

Also: fix the two settings mismatches surfaced during verification —
`INGESTION_MAX_FILE_SIZE_BYTES` defaults to 200 MB despite a `# 10 MB` comment
(`settings.py:217`), and `INGESTION_REQUIRE_AV` defaults false; production profile should
set an explicit cap and require AV.

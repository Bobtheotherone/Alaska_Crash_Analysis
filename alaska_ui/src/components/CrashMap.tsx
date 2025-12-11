import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  CircleMarker,
  MapContainer,
  Popup,
  Rectangle,
  TileLayer,
  useMapEvents,
} from "react-leaflet";
import type { LatLngBounds, LatLngBoundsExpression } from "leaflet";
import "leaflet/dist/leaflet.css";

type IngestionRowChecks = {
  total_rows: number;
  invalid_row_count: number;
  invalid_geo_row_count: number;
};

interface CrashMapProps {
  uploadId: string | null;
  authHeader: () => Record<string, string>;
  initialSeverity?: string[];
  initialMunicipality?: string;
  initialStartDate?: string;
  initialEndDate?: string;
  ingestionRowChecks?: IngestionRowChecks;
}

type CrashFeature = {
  type: "Feature";
  geometry: { type: "Point"; coordinates: [number, number] };
  properties: {
    id: number;
    crash_id: string;
    severity: string;
    crash_datetime: string;
    roadway_name: string;
    municipality: string;
    posted_speed_limit: number | null;
    dataset_id: string;
  };
};

type HeatCell = {
  count: number;
  center: [number, number];
  grid_size: number;
};

type Bbox = {
  minLon: number;
  minLat: number;
  maxLon: number;
  maxLat: number;
};

const ALASKA_BOUNDS: LatLngBoundsExpression = [
  [51.20978, -179.14734],
  [71.38957, -129.97955],
];

const DEFAULT_CENTER: [number, number] = [63.5, -150];

const POINT_LIMIT = 5000;
const FETCH_DEBOUNCE_MS = 250;

const SEVERITY_COLORS: Record<string, string> = {
  K: "#b91c1c",
  A: "#fb923c",
  B: "#f59e0b",
  C: "#fcd34d",
  O: "#6b7280",
};

const boundsToBbox = (bounds: LatLngBounds): Bbox => ({
  minLon: bounds.getWest(),
  minLat: bounds.getSouth(),
  maxLon: bounds.getEast(),
  maxLat: bounds.getNorth(),
});

const gridSizeForZoom = (zoom: number) => {
  if (zoom >= 11) return 0.01;
  if (zoom >= 8) return 0.025;
  if (zoom >= 6) return 0.05;
  if (zoom >= 5) return 0.1;
  return 0.25;
};

const getHeatColor = (count: number, maxCount: number) => {
  if (maxCount <= 0) return "#e5e7eb";
  const t = Math.min(1, count / maxCount);
  const r = Math.round(234 + (190 - 234) * (1 - t));
  const g = Math.round(88 + (24 - 88) * t);
  const b = Math.round(12 + (32 - 12) * t);
  return `rgba(${r}, ${g}, ${b}, ${0.35 + 0.45 * t})`;
};

const MapEventWatcher: React.FC<{
  onViewportChange: (bbox: Bbox, zoom: number) => void;
}> = ({ onViewportChange }) => {
  const map = useMapEvents({
    moveend: () => onViewportChange(boundsToBbox(map.getBounds()), map.getZoom()),
    zoomend: () => onViewportChange(boundsToBbox(map.getBounds()), map.getZoom()),
  });

  useEffect(() => {
    const initialBounds = map.getBounds();
    onViewportChange(boundsToBbox(initialBounds), map.getZoom());
  }, [map, onViewportChange]);

  return null;
};

const CrashMap: React.FC<CrashMapProps> = ({
  uploadId,
  authHeader,
  initialSeverity = [],
  initialMunicipality = "",
  initialStartDate = "",
  initialEndDate = "",
  ingestionRowChecks,
}) => {
  const [bbox, setBbox] = useState<Bbox | null>(null);
  const [zoom, setZoom] = useState<number>(5);
  const [severity, setSeverity] = useState<string[]>(initialSeverity);
  const [municipality, setMunicipality] = useState(initialMunicipality || "");
  const [startDate, setStartDate] = useState<string | null>(
    initialStartDate || null
  );
  const [endDate, setEndDate] = useState<string | null>(
    initialEndDate || null
  );
  const [showPoints, setShowPoints] = useState(true);
  const [showHeatmap, setShowHeatmap] = useState(false);

  const [features, setFeatures] = useState<CrashFeature[]>([]);
  const [featureCount, setFeatureCount] = useState(0);
  const [heatCells, setHeatCells] = useState<HeatCell[]>([]);

  const [pointLoading, setPointLoading] = useState(false);
  const [heatLoading, setHeatLoading] = useState(false);
  const [pointError, setPointError] = useState<string | null>(null);
  const [heatError, setHeatError] = useState<string | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [limitNotice, setLimitNotice] = useState(false);
  const [limitValue, setLimitValue] = useState<number | null>(null);
  const [dateError, setDateError] = useState<string | null>(null);

  const pointAbortRef = useRef<AbortController | null>(null);
  const heatAbortRef = useRef<AbortController | null>(null);

  const gridSize = useMemo(() => gridSizeForZoom(zoom), [zoom]);

  const onViewportChange = useCallback((nextBbox: Bbox, nextZoom: number) => {
    setBbox(nextBbox);
    setZoom(nextZoom);
  }, []);

  const toggleSeverity = (code: string) => {
    setSeverity((prev) =>
      prev.includes(code) ? prev.filter((s) => s !== code) : [...prev, code]
    );
  };

  const appendFilters = (params: URLSearchParams) => {
    if (severity.length) {
      params.set("severity", severity.join(","));
    }
    if (municipality.trim()) {
      params.set("municipality", municipality.trim());
    }
    if (startDate && startDate.trim()) {
      params.set("start_datetime", startDate.trim());
    }
    if (endDate && endDate.trim()) {
      params.set("end_datetime", endDate.trim());
    }
  };

  useEffect(() => {
    setDateError(null);
    if (startDate && endDate) {
      const startMs = Date.parse(startDate);
      const endMs = Date.parse(endDate);
      if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) {
        setDateError("Invalid date/time format.");
      } else if (startMs > endMs) {
        setDateError("Start date/time must be before or equal to end date/time.");
      }
    }
  }, [startDate, endDate]);

  useEffect(() => {
    if (!uploadId || !bbox || !showPoints) {
      setFeatures([]);
      setFeatureCount(0);
      setLimitNotice(false);
      setPointLoading(false);
      setPointError(null);
      setLimitValue(null);
      return;
    }

    if (dateError) {
      setPointError(dateError);
      setPointLoading(false);
      return;
    }

    if (pointAbortRef.current) {
      pointAbortRef.current.abort();
    }

    const controller = new AbortController();
    pointAbortRef.current = controller;
    setPointLoading(true);
    setPointError(null);

    const timer = setTimeout(async () => {
      try {
        const params = new URLSearchParams({
          upload_id: uploadId,
          min_lon: bbox.minLon.toString(),
          min_lat: bbox.minLat.toString(),
          max_lon: bbox.maxLon.toString(),
          max_lat: bbox.maxLat.toString(),
          limit: POINT_LIMIT.toString(),
        });
        appendFilters(params);

        const resp = await fetch(
          `/api/crashdata/crashes-within-bbox/?${params.toString()}`,
          {
            method: "GET",
            headers: { Accept: "application/json", ...authHeader() },
            signal: controller.signal,
          }
        );

        if (!resp.ok) {
          let message = `Request failed (${resp.status})`;
          if (resp.status === 401 || resp.status === 403) {
            message =
              "You may not be logged in or lack access to this dataset.";
          } else if (resp.status === 400) {
            try {
              const data = await resp.json();
              if (data?.detail) message = String(data.detail);
            } catch {
              // ignore parse errors
            }
          }
          throw new Error(message);
        }

        const data = await resp.json();
        const respFeatures = (data?.features as CrashFeature[]) || [];
        const respCount = Number(data?.count ?? respFeatures.length);
        const respLimit = Number(data?.limit ?? 0);
        setFeatures(respFeatures);
        setFeatureCount(respCount);
        setLimitNotice(respLimit > 0 && respCount === respLimit);
        setLimitValue(respLimit > 0 ? respLimit : null);
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        console.error("[CrashMap] Point fetch failed:", err);
        setPointError(
          err instanceof Error
            ? err.message
            : "Could not load crash points for this view."
        );
      } finally {
        setPointLoading(false);
      }
    }, FETCH_DEBOUNCE_MS);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [
    uploadId,
    bbox,
    severity,
    municipality,
    startDate,
    endDate,
    showPoints,
    authHeader,
    dateError,
  ]);

  useEffect(() => {
    if (!uploadId || !bbox || !showHeatmap) {
      setHeatCells([]);
      setHeatLoading(false);
      setHeatError(null);
      return;
    }

    if (dateError) {
      setHeatError(dateError);
      setHeatLoading(false);
      return;
    }

    if (heatAbortRef.current) {
      heatAbortRef.current.abort();
    }

    const controller = new AbortController();
    heatAbortRef.current = controller;
    setHeatLoading(true);
    setHeatError(null);

    const timer = setTimeout(async () => {
      try {
        const params = new URLSearchParams({
          upload_id: uploadId,
          min_lon: bbox.minLon.toString(),
          min_lat: bbox.minLat.toString(),
          max_lon: bbox.maxLon.toString(),
          max_lat: bbox.maxLat.toString(),
          grid_size: gridSize.toString(),
        });
        appendFilters(params);

        const resp = await fetch(
          `/api/crashdata/heatmap/?${params.toString()}`,
          {
            method: "GET",
            headers: { Accept: "application/json", ...authHeader() },
            signal: controller.signal,
          }
        );

        if (!resp.ok) {
          let message = `Request failed (${resp.status})`;
          if (resp.status === 401 || resp.status === 403) {
            message =
              "You may not be logged in or lack access to this dataset.";
          } else if (resp.status === 400) {
            try {
              const data = await resp.json();
              if (data?.detail) message = String(data.detail);
            } catch {
              // ignore parse errors
            }
          }
          throw new Error(message);
        }

        const data = await resp.json();
        setHeatCells((data?.cells as HeatCell[]) || []);
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        console.error("[CrashMap] Heatmap fetch failed:", err);
        setHeatError(
          err instanceof Error
            ? err.message
            : "Could not load crash density for this view."
        );
      } finally {
        setHeatLoading(false);
      }
    }, FETCH_DEBOUNCE_MS);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [
    uploadId,
    bbox,
    severity,
    municipality,
    startDate,
    endDate,
    showHeatmap,
    gridSize,
    authHeader,
    dateError,
  ]);

  useEffect(() => {
    setExportError(null);
  }, [uploadId, severity, municipality, startDate, endDate]);

  const handleExport = async () => {
    if (!uploadId) return;
    setExportError(null);

    const params = new URLSearchParams({ upload_id: uploadId });
    appendFilters(params);

    try {
      const resp = await fetch(
        `/api/crashdata/exports/crashes.csv?${params.toString()}`,
        {
          method: "GET",
          headers: { ...authHeader() },
        }
      );

      if (!resp.ok) {
        let message = `Export failed (${resp.status})`;
        if (resp.status === 401 || resp.status === 403) {
          message = "You may not be logged in or lack access to this dataset.";
        } else {
          try {
            const data = await resp.json();
            if (data?.detail) {
              message = String(data.detail);
              if (data?.row_count && data?.max_rows) {
                message += ` (row_count=${data.row_count}, max_rows=${data.max_rows})`;
              }
            }
          } catch {
            // ignore parse errors
          }
        }
        throw new Error(message);
      }

      const blob = await resp.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `crashes_${uploadId}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error("[CrashMap] Export failed:", err);
      setExportError(
        err instanceof Error
          ? err.message
          : "Could not export filtered crashes. Please try again."
      );
    }
  };

  const ingestionSummary = useMemo(() => {
    if (!ingestionRowChecks) return null;
    const { total_rows, invalid_row_count, invalid_geo_row_count } =
      ingestionRowChecks;
    const total = total_rows?.toLocaleString?.() ?? 0;
    const invalid = invalid_row_count?.toLocaleString?.() ?? 0;
    const invalidGeo = invalid_geo_row_count?.toLocaleString?.() ?? 0;
    return {
      text: `${total} rows, ${invalid} invalid, ${invalidGeo} invalid coordinates (excluded from map).`,
      hasGeoIssues: (ingestionRowChecks.invalid_geo_row_count || 0) > 0,
    };
  }, [ingestionRowChecks]);

  const maxHeatCount = useMemo(
    () => heatCells.reduce((max, cell) => Math.max(max, cell.count), 0),
    [heatCells]
  );

  useEffect(() => {
    if (ingestionRowChecks && ingestionRowChecks.invalid_geo_row_count > 0) {
      console.warn(
        "[CrashMap] Some rows have invalid/out-of-bounds coordinates and will be excluded."
      );
    }
  }, [ingestionRowChecks]);

  if (!uploadId) {
    return (
      <div className="h-full bg-neutral-light border border-neutral-medium rounded-lg p-6 flex flex-col items-center justify-center text-center">
        <h3 className="text-lg font-semibold text-neutral-darker mb-2">
          No dataset selected
        </h3>
        <p className="text-sm text-neutral-darker max-w-xl">
          Import and validate a dataset, then run the ETL to see crashes on the
          map. Crash markers and density will appear here once an upload is
          available.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4 h-full">
      <div className="flex flex-col gap-2 bg-neutral-light border border-neutral-medium rounded-md p-4">
        <div className="flex flex-wrap items-center gap-3 justify-between">
          <div className="text-sm text-neutral-darker">
            {ingestionSummary ? (
              <span
                className={`font-medium ${ingestionSummary.hasGeoIssues ? "text-amber-700" : "text-neutral-darker"
                  }`}
              >
                Data quality: {ingestionSummary.text}
              </span>
            ) : (
              <span className="text-neutral-darker">
                Data quality: Ingestion stats not available yet.
              </span>
            )}
          </div>
          <button
            type="button"
            onClick={handleExport}
            className="inline-flex items-center rounded-md border border-neutral-medium bg-white px-3 py-1.5 text-xs font-semibold text-neutral-darker hover:bg-neutral-light"
          >
            Export map-filtered crashes (CSV)
          </button>
        </div>
        {ingestionSummary?.hasGeoIssues && (
          <div className="flex items-center gap-2 text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded px-3 py-2">
            <span className="font-semibold">Heads up:</span>
            <span>
              Some rows have invalid or out-of-bounds coordinates. Only crashes
              with valid locations are shown on the map.
            </span>
          </div>
        )}
        {exportError && (
          <div className="text-xs text-red-700 bg-red-50 border border-red-200 rounded px-3 py-2">
            {exportError}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-2">
        <div className="flex flex-col gap-2 lg:col-span-2">
          <div className="flex flex-wrap items-center gap-2">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={showPoints}
                onChange={(e) => setShowPoints(e.target.checked)}
              />
              Crash points
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={showHeatmap}
                onChange={(e) => setShowHeatmap(e.target.checked)}
              />
              Crash density
            </label>
            <div className="flex items-center gap-2 text-xs text-neutral-darker">
              <span className="font-semibold">Legend:</span>
              {Object.entries(SEVERITY_COLORS).map(([code, color]) => (
                <span key={code} className="flex items-center gap-1">
                  <span
                    className="inline-block rounded-full"
                    style={{
                      width: "10px",
                      height: "10px",
                      backgroundColor: color,
                    }}
                  />
                  {code}
                </span>
              ))}
            </div>
          </div>
          <div className="flex flex-wrap gap-2 text-sm">
            {["K", "A", "B", "C", "O"].map((code) => (
              <button
                key={code}
                type="button"
                onClick={() => toggleSeverity(code)}
                className={`px-2 py-1 rounded border text-xs ${severity.includes(code)
                    ? "bg-brand-primary text-white border-brand-primary"
                    : "bg-white text-neutral-darker border-neutral-medium"
                  }`}
              >
                {code}
              </button>
            ))}
            <button
              type="button"
              onClick={() => setSeverity([])}
              className="px-2 py-1 rounded border border-neutral-medium text-xs text-neutral-darker bg-white"
            >
              Clear severity
            </button>
          </div>
        </div>

        <div className="flex flex-wrap lg:justify-end gap-3 text-sm items-end">
          <div className="flex flex-col">
            <label className="text-xs text-neutral-darker mb-1">
              Municipality
            </label>
            <input
              type="text"
              value={municipality}
              onChange={(e) => setMunicipality(e.target.value)}
              className="border border-neutral-medium rounded px-2 py-1 text-sm w-48"
              placeholder="Municipality (optional)"
            />
          </div>
          <div className="flex flex-col">
            <label className="text-xs text-neutral-darker mb-1">
              Start date / time
            </label>
            <input
              type="datetime-local"
              value={startDate ?? ""}
              onChange={(e) => setStartDate(e.target.value || null)}
              className="border border-neutral-medium rounded px-2 py-1 text-sm w-48"
              placeholder="mm/dd/yyyy --:-- --"
            />
          </div>
          <div className="flex flex-col">
            <label className="text-xs text-neutral-darker mb-1">
              End date / time
            </label>
            <input
              type="datetime-local"
              value={endDate ?? ""}
              onChange={(e) => setEndDate(e.target.value || null)}
              className="border border-neutral-medium rounded px-2 py-1 text-sm w-48"
              placeholder="mm/dd/yyyy --:-- --"
            />
          </div>
        </div>
      </div>
      <div className="text-xs text-neutral-darker">
        Leave blank to include all crash dates.
      </div>
      {dateError && (
        <div className="text-xs text-red-700 bg-red-50 border border-red-200 rounded px-3 py-2">
          {dateError}
        </div>
      )}

      <div className="flex-1 min-h-[420px] rounded-lg overflow-hidden border border-neutral-medium bg-white shadow-sm">
        <MapContainer
          bounds={ALASKA_BOUNDS}
          center={DEFAULT_CENTER}
          zoom={4}
          maxBounds={ALASKA_BOUNDS}
          style={{ height: "100%", width: "100%" }}
          scrollWheelZoom
          whenCreated={(mapInstance) => {
            const initialBounds = mapInstance.getBounds();
            setBbox(boundsToBbox(initialBounds));
            setZoom(mapInstance.getZoom());
          }}
        >
          <TileLayer
            attribution='&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <MapEventWatcher onViewportChange={onViewportChange} />

          {showHeatmap &&
            heatCells.map((cell, idx) => {
              const half = cell.grid_size / 2;
              const lat = cell.center[1];
              const lon = cell.center[0];
              const bounds: LatLngBoundsExpression = [
                [lat - half, lon - half],
                [lat + half, lon + half],
              ];
              return (
                <Rectangle
                  key={`${idx}-${cell.center.join(",")}`}
                  bounds={bounds}
                  pathOptions={{
                    fillColor: getHeatColor(cell.count, maxHeatCount),
                    color: getHeatColor(cell.count, maxHeatCount),
                    weight: 1,
                    fillOpacity: 0.6,
                  }}
                />
              );
            })}

          {showPoints &&
            features.map((feature) => {
              const [lon, lat] = feature.geometry.coordinates;
              const sev = (feature.properties.severity || "").toUpperCase();
              const color = SEVERITY_COLORS[sev] || "#2563eb";
              return (
                <CircleMarker
                  key={feature.properties.id}
                  center={[lat, lon]}
                  radius={7}
                  pathOptions={{ color, fillColor: color, fillOpacity: 0.85 }}
                >
                  <Popup>
                    <div className="text-sm">
                      <div className="font-semibold mb-1">
                        Crash {feature.properties.crash_id || feature.properties.id}
                      </div>
                      <div className="space-y-1 text-xs">
                        <div>
                          <span className="font-semibold">Severity:</span>{" "}
                          {feature.properties.severity}
                        </div>
                        <div>
                          <span className="font-semibold">Date/time:</span>{" "}
                          {feature.properties.crash_datetime}
                        </div>
                        <div>
                          <span className="font-semibold">Roadway:</span>{" "}
                          {feature.properties.roadway_name || "Unknown"}
                        </div>
                        <div>
                          <span className="font-semibold">Municipality:</span>{" "}
                          {feature.properties.municipality || "Unknown"}
                        </div>
                        <div>
                          <span className="font-semibold">Speed limit:</span>{" "}
                          {feature.properties.posted_speed_limit ?? "n/a"}
                        </div>
                      </div>
                    </div>
                  </Popup>
                </CircleMarker>
              );
            })}
        </MapContainer>
      </div>

      <div className="flex flex-wrap gap-3 text-xs text-neutral-darker">
        <div className="flex items-center gap-2">
          {pointLoading ? (
            <span className="font-semibold text-brand-primary">
              Loading crash points…
            </span>
          ) : (
            <span>
              Showing {featureCount.toLocaleString()} crash
              {featureCount === 1 ? "" : "es"} in view
            </span>
          )}
          {limitNotice && (
            <span className="text-amber-800 bg-amber-50 border border-amber-200 rounded px-2 py-1">
              Showing {limitValue ?? POINT_LIMIT} crashes (server limit). Zoom
              or filter to see more detail.
            </span>
          )}
        </div>
        {showHeatmap && (
          <div className="flex items-center gap-2">
            {heatLoading ? (
              <span className="text-brand-primary font-semibold">
                Loading density…
              </span>
            ) : (
              <span>
                Density grid size {gridSize}° · {heatCells.length} cells
              </span>
            )}
          </div>
        )}
        {!pointLoading && !pointError && features.length === 0 && (
          <span className="text-neutral-darker bg-neutral-light border border-neutral-medium rounded px-2 py-1">
            No crashes in this view for the current filters.
          </span>
        )}
        {showHeatmap && !heatLoading && !heatError && heatCells.length === 0 && (
          <span className="text-neutral-darker bg-neutral-light border border-neutral-medium rounded px-2 py-1">
            No density cells for this view and filters.
          </span>
        )}
        {pointError && (
          <span className="text-red-700 bg-red-50 border border-red-200 rounded px-2 py-1">
            {pointError}
          </span>
        )}
        {heatError && (
          <span className="text-red-700 bg-red-50 border border-red-200 rounded px-2 py-1">
            {heatError}
          </span>
        )}
      </div>
    </div>
  );
};

export default CrashMap;

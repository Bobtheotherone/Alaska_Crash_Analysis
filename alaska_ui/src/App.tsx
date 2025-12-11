import React, { useState, useCallback, useEffect } from "react";
import WorkflowPanel from "./components/WorkflowPanel";
import MainContent from "./components/MainContent";
import { UserCircleIcon } from "./constants";
import { runValidationLogic } from "./lib/validator";

// Globals loaded from script tags in index.html
declare const Papa: any;
declare const XLSX: any;
declare const Chart: any;

export interface DataPrepState {
  unknownThreshold: number;
  yesNoThreshold: number;
  speedLimit: number;
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

  // Manual non-leakage drops to align UI + backend cleaning.
  columnsToDrop: string[];

  // User override for which column is treated as severity/outcome.
  severityColumn: string | null;

  // Extra tokens to treat as "unknown" beyond the built-in list.
  additionalUnknownTokens: string[];
}

export interface ColumnStat {
  column: string;
  unknownPercent: number;
  yesNoStats: {
    yesPercent: number;
    noPercent: number;
    totalYesNo: number;
    coveragePercent: number;
  } | null;
  status: "Keep" | "Drop";
  reason: string | null;
}

export interface ValidationResults {
  rowCount: number;
  columnCount: number;
  droppedColumnCount: number;
  columnStats: ColumnStat[];
  error?: string;

  // Ingestion gateway / upload-safety summary returned by the backend.
  // These fields power the "Ingestion gateway: accepted/rejected" banner and the ingestion checks log.
  ingestionOverallStatus?: "accepted" | "rejected" | "pending" | string;
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
    status: "passed" | "failed" | "skipped";
    severity?: "info" | "warning" | "error";
    code?: string;
    is_hard_fail?: boolean;
  }[];
}


export interface ClassificationReportRow {
  className: string;
  precision: number;
  recall: number;
  "f1-score": number;
  support: number;
}

export interface AnalysisResults {
  featureImportance: { feature: string; importance: number }[];
  decisionRules: string[];
  classificationReport?: ClassificationReportRow[];
  confusionMatrix?: number[][];
  classLabels?: string[];
  error?: string;
}

interface AuthState {
  username: string;
  password: string;
  isLoggedIn: boolean;
  error: string | null;
}

// Explicit stages for the import/validation pipeline
export type ValidationStage =
  | "idle"
  | "parsing"
  | "local_validation"
  | "uploading"
  | "complete"
  | "error";


// -------- Ingestion / upload-safety helpers --------

// Normalize unknown/enum-ish values from the backend to a consistent lowercase string.
const toLower = (value: unknown): string | undefined => {
  if (value === null || value === undefined) return undefined;
  const s = String(value).trim();
  if (!s) return undefined;
  return s.toLowerCase();
};

const asNumber = (value: unknown): number | undefined => {
  if (value === null || value === undefined || value === "") return undefined;
  const n = Number(value);
  return Number.isFinite(n) ? n : undefined;
};

const normalizeIngestionOverallStatus = (
  raw: unknown
): ValidationResults["ingestionOverallStatus"] | undefined => {
  const s = toLower(raw);
  if (!s) return undefined;

  if (
    [
      "accepted",
      "accept",
      "ok",
      "passed",
      "pass",
      "success",
      "succeeded",
      "safe",
    ].includes(s)
  ) {
    return "accepted";
  }

  if (
    [
      "rejected",
      "reject",
      "failed",
      "fail",
      "error",
      "unsafe",
      "denied",
      "blocked",
    ].includes(s)
  ) {
    return "rejected";
  }

  if (
    [
      "pending",
      "processing",
      "in_progress",
      "in-progress",
      "queued",
      "running",
    ].includes(s)
  ) {
    return "pending";
  }

  // Unknown / unmapped status from backend
  return "unknown";
};

const normalizeIngestionStepStatus = (
  raw: unknown
): "passed" | "failed" | "skipped" => {
  const s = toLower(raw);
  if (!s) return "passed";

  if (["failed", "fail", "error", "rejected", "deny", "denied"].includes(s)) {
    return "failed";
  }

  if (["skipped", "skip", "ignored", "n/a", "na"].includes(s)) {
    return "skipped";
  }

  return "passed";
};

const normalizeIngestionSeverity = (
  raw: unknown
): "info" | "warning" | "error" | undefined => {
  const s = toLower(raw);
  if (!s) return undefined;

  if (["error", "err", "fatal", "critical"].includes(s)) return "error";
  if (["warning", "warn"].includes(s)) return "warning";
  if (["info", "information", "debug"].includes(s)) return "info";
  return undefined;
};

/**
 * Best-effort extraction of ingestion-gateway summary fields from the backend upload response.
 * The backend's JSON keys have changed a few times, so this function supports multiple aliases.
 */
export const extractIngestionSummary = (
  backendSummary: any
): Pick<
  ValidationResults,
  | "ingestionOverallStatus"
  | "ingestionMessage"
  | "ingestionErrorCode"
  | "ingestionRowChecks"
  | "ingestionSteps"
> => {
  const summary = backendSummary ?? {};

  const ingestionOverallStatus = normalizeIngestionOverallStatus(
    summary.ingestion_overall_status ??
      summary.ingestionOverallStatus ??
      summary.overall_status ??
      summary.overallStatus ??
      summary.ingestion_status ??
      summary.ingestionStatus ??
      summary.gateway_status ??
      summary.gatewayStatus ??
      summary.status
  );

  const ingestionMessage: string | undefined =
    summary.ingestion_message ??
    summary.ingestionMessage ??
    summary.message ??
    summary.detail ??
    summary.details;

  const ingestionErrorCode: string | undefined =
    summary.ingestion_error_code ??
    summary.ingestionErrorCode ??
    summary.error_code ??
    summary.errorCode ??
    summary.code;

  // Row counts may be returned as a nested object or as top-level fields.
  const rowChecksRaw =
    summary.ingestion_row_checks ??
    summary.ingestionRowChecks ??
    summary.row_checks ??
    summary.rowChecks ??
    summary.rows;

  const totalRows =
    asNumber(rowChecksRaw?.total_rows) ??
    asNumber(rowChecksRaw?.rows_seen) ??
    asNumber(rowChecksRaw?.rowsSeen) ??
    asNumber(summary.total_rows) ??
    asNumber(summary.rows_seen) ??
    asNumber(summary.rowsSeen);

  const invalidRowCount =
    asNumber(rowChecksRaw?.invalid_row_count) ??
    asNumber(rowChecksRaw?.invalid_rows) ??
    asNumber(rowChecksRaw?.invalid_schema_row_count) ??
    asNumber(rowChecksRaw?.invalid_schema_rows) ??
    asNumber(rowChecksRaw?.invalid_value_row_count) ??
    asNumber(summary.invalid_row_count) ??
    asNumber(summary.invalid_rows) ??
    asNumber(summary.invalid_schema_rows) ??
    asNumber(summary.invalid_value_rows);

  const invalidGeoRowCount =
    asNumber(rowChecksRaw?.invalid_geo_row_count) ??
    asNumber(rowChecksRaw?.invalid_geometry_row_count) ??
    asNumber(rowChecksRaw?.invalid_geometry_rows) ??
    asNumber(rowChecksRaw?.invalid_geo_rows) ??
    asNumber(summary.invalid_geo_row_count) ??
    asNumber(summary.invalid_geometry_rows) ??
    asNumber(summary.invalid_geo_rows);

  const ingestionRowChecks =
    totalRows !== undefined ||
    invalidRowCount !== undefined ||
    invalidGeoRowCount !== undefined
      ? {
          total_rows: totalRows ?? 0,
          invalid_row_count: invalidRowCount ?? 0,
          invalid_geo_row_count: invalidGeoRowCount ?? 0,
        }
      : undefined;

  // Ingestion check list / log.
  const stepsRaw =
    summary.ingestion_steps ??
    summary.ingestionSteps ??
    summary.steps ??
    summary.checks ??
    summary.ingestion_checks ??
    summary.ingestionChecks;

  const ingestionSteps = Array.isArray(stepsRaw)
    ? (stepsRaw
        .map((s: any) => {
          const step = String(
            s.step ?? s.check ?? s.name ?? s.id ?? s.type ?? ""
          ).trim();
          if (!step) return null;

          const message = String(
            s.details ?? s.message ?? s.detail ?? s.info ?? s.reason ?? ""
          ).trim();

          const status = normalizeIngestionStepStatus(
            s.status ?? s.result ?? s.outcome
          );
          const severity =
            normalizeIngestionSeverity(s.severity ?? s.level ?? s.kind) ??
            (status === "passed" ? "info" : undefined);
          const code = s.code ?? s.error_code ?? s.errorCode;
          const is_hard_fail = Boolean(
            s.is_hard_fail ?? s.hard_fail ?? s.isHardFail ?? false
          );

          return {
            step,
            message,
            status,
            ...(severity ? { severity } : {}),
            ...(code ? { code: String(code) } : {}),
            is_hard_fail,
          };
        })
        .filter(Boolean) as NonNullable<ValidationResults["ingestionSteps"]>)
    : undefined;

  return {
    ...(ingestionOverallStatus ? { ingestionOverallStatus } : {}),
    ...(ingestionMessage ? { ingestionMessage } : {}),
    ...(ingestionErrorCode ? { ingestionErrorCode } : {}),
    ...(ingestionRowChecks ? { ingestionRowChecks } : {}),
    ...(ingestionSteps ? { ingestionSteps } : {}),
  };
};

// Helper to format a short error string
const formatShortError = (msg: string | undefined): string =>
  msg && msg.length > 80 ? msg.slice(0, 80) + "..." : msg || "";

// Detection of unknown-ish tokens (client-side validator)
const BASE_UNKNOWN_TOKENS = new Set([
  "no data",
  "missing value",
  "null value",
  "missing",
  "na",
  "n/a",
  "n.a.",
  "none",
  "null",
  "nan",
  "unknown",
  "unspecified",
  "not specified",
  "not applicable",
  "tbd",
  "tba",
  "to be determined",
  "-",
  "--",
  "(blank)",
  "blank",
  "(null)",
  "?",
  "prefer not to say",
  "refused",
]);

const UNKNOWN_SUBSTRINGS = new Set([
  "unknown",
  "missing",
  "unspecified",
  "not specified",
  "not applicable",
  "n/a",
  "na",
  "null",
  "blank",
  "tbd",
  "tba",
  "to be determined",
  "refused",
  "prefer not to say",
  "no data",
  "no value",
]);

const YES_TOKENS = new Set(["yes", "y", "true", "t"]);
const NO_TOKENS = new Set(["no", "n", "false", "f"]);

// Toggle this to true when you want the App version logger
const APP_VERSION_LOG_ENABLED = false;

function discoverUnknownTokens(
  rows: Record<string, unknown>[],
  extraUnknowns: string[] = []
): Set<string> {
  const counts = new Map<string, number>();
  const patterns = Array.from(UNKNOWN_SUBSTRINGS).map(
    (tok) =>
      new RegExp(`\\b${tok.replace(/[-/\\^$*+?.()|[\]{}]/g, "\\$&")}\\b`, "i")
  );

  const normalizedExtras = extraUnknowns
    .map((tok) => tok.trim().toLowerCase())
    .filter((tok) => !!tok);
  const seedUnknowns = new Set<string>([
    ...BASE_UNKNOWN_TOKENS,
    ...normalizedExtras,
  ]);

  if (!rows.length) return seedUnknowns;

  const columns = Object.keys(rows[0]);

  for (const col of columns) {
    for (const row of rows) {
      const raw = row[col];
      if (raw == null) continue;

      const norm = String(raw).trim().toLowerCase();
      if (!norm || norm.length > 80) continue;

      if (seedUnknowns.has(norm)) {
        counts.set(norm, (counts.get(norm) || 0) + 1);
        continue;
      }

      if (patterns.some((re) => re.test(norm))) {
        counts.set(norm, (counts.get(norm) || 0) + 1);
      }
    }
  }

  const discovered = new Set<string>();
  for (const [tok, count] of counts.entries()) {
    if (count >= 2) discovered.add(tok);
  }

  return new Set([...seedUnknowns, ...discovered]);
}

// ---------------------------------------------------------------------------
// Data-leakage helpers (UI-side, Peyton-style)
// ---------------------------------------------------------------------------

const LEAKAGE_KEYWORDS = [
  "fatal",
  "fatalities",
  "death",
  "dead",
  "killed",
  "injury",
  "injuries",
  "injured",
  "severity",
  "severe",
  "serious",
  "k_count",
  "killed_cnt",
  "inj_cnt",
];

function suggestLeakageColumnsByName(columns: string[]): string[] {
  const suggestions = new Set<string>();

  for (const col of columns) {
    const lower = col.toLowerCase();
    if (LEAKAGE_KEYWORDS.some((kw) => lower.includes(kw))) {
      suggestions.add(col);
    }
  }

  return Array.from(suggestions).sort();
}

/**
 * Given validation results and a set of leakage columns, forcibly mark
 * those columns as Drop with a clear explanation. Mirrors Peyton's
 * behaviour of always removing user-marked leakage columns.
 */
function applyColumnOverrides(
  results: ValidationResults,
  opts: {
    leakageColumns?: string[];
    manualDropColumns?: string[];
  }
): ValidationResults {
  const leakSet = new Set(opts.leakageColumns || []);
  const manualSet = new Set(opts.manualDropColumns || []);

  if (leakSet.size === 0 && manualSet.size === 0) {
    return results;
  }

  const columnStats = results.columnStats.map((stat) => {
    const reasons: string[] = [];
    let shouldDrop = stat.status === "Drop";

    if (leakSet.has(stat.column)) {
      shouldDrop = true;
      reasons.push(
        "Flagged as a data-leakage column (for example it may contain final injury/fatality counts or another severity code)."
      );
    }

    if (manualSet.has(stat.column)) {
      shouldDrop = true;
      reasons.push("User-marked as low-value (manual drop).");
    }

    if (!shouldDrop) {
      return stat;
    }

    const combinedReason = [stat.reason, reasons.join(" ")].filter(Boolean).join(" ");

    return {
      ...stat,
      status: "Drop" as const,
      reason: combinedReason || null,
    };
  });

  const droppedColumnCount = columnStats.filter(
    (cs) => cs.status === "Drop"
  ).length;

  return {
    ...results,
    droppedColumnCount,
    columnStats,
  };
}

function runClientValidation(
  rows: Record<string, string>[],
  config: DataPrepState
): ValidationResults {
  const rowCount = rows.length;
  if (!rowCount) {
    return {
      rowCount: 0,
      columnCount: 0,
      droppedColumnCount: 0,
      columnStats: [],
    };
  }

  const columns = Object.keys(rows[0]);
  const colCount = columns.length;
  const unknownTokens = discoverUnknownTokens(
    rows,
    config.additionalUnknownTokens || []
  );

  const stats: ColumnStat[] = columns.map((col) => {
    let unknownCount = 0;
    let yesCount = 0;
    let noCount = 0;
    const yesNoValues: string[] = [];

    for (const row of rows) {
      const raw = row[col];
      if (raw == null || raw === "") {
        unknownCount++;
        continue;
      }
      const norm = String(raw).trim().toLowerCase();
      if (unknownTokens.has(norm)) {
        unknownCount++;
        continue;
      }

      yesNoValues.push(norm);
    }

    for (const v of yesNoValues) {
      if (YES_TOKENS.has(v)) yesCount++;
      else if (NO_TOKENS.has(v)) noCount++;
    }

    const unknownPercent = (unknownCount / rowCount) * 100;
    const yesNoTotal = yesCount + noCount;
    const yesNoCoveragePercent = yesNoTotal ? (yesNoTotal / rowCount) * 100 : 0;

    let yesNoStats: ColumnStat["yesNoStats"] = null;
    if (yesNoTotal > 0) {
      yesNoStats = {
        yesPercent: (yesCount / yesNoTotal) * 100,
        noPercent: (noCount / yesNoTotal) * 100,
        totalYesNo: yesNoTotal,
        coveragePercent: yesNoCoveragePercent,
      };
    }

    let status: ColumnStat["status"] = "Keep";
    let reason: string | null = null;

    if (unknownPercent > config.unknownThreshold) {
      status = "Drop";
      reason = `Exceeds unknown threshold (${unknownPercent.toFixed(
        1
      )}% > ${config.unknownThreshold}%)`;
    } else if (
      yesNoStats &&
      yesNoCoveragePercent >= 50 &&
      (yesNoStats.yesPercent < config.yesNoThreshold ||
        yesNoStats.noPercent < config.yesNoThreshold)
    ) {
      status = "Drop";
      reason = `Extreme Yes/No imbalance (Yes: ${yesNoStats.yesPercent.toFixed(
        1
      )}%, No: ${yesNoStats.noPercent.toFixed(1)}%)`;
    }

    return {
      column: col,
      unknownPercent,
      yesNoStats,
      status,
      reason,
    };
  });

  const droppedColumnCount = stats.filter((s) => s.status === "Drop").length;

  return {
    rowCount,
    columnCount: colCount,
    droppedColumnCount,
    columnStats: stats,
  };
}

// ---------- Login screen component ----------

interface LoginScreenProps {
  auth: AuthState;
  onFieldChange: (key: "username" | "password", value: string) => void;
  onLogin: () => void;
}

const LoginScreen: React.FC<LoginScreenProps> = ({
  auth,
  onFieldChange,
  onLogin,
}) => {
  return (
    <div className="min-h-screen bg-neutral-light flex items-center justify-center">
      <div className="bg-white shadow-lg rounded-lg p-8 max-w-sm w-full border border-neutral-medium">
        <div className="flex items-center justify-center mb-4">
          <UserCircleIcon />
        </div>
        <h1 className="text-xl font-semibold text-center text-brand-primary mb-4">
          Sign in to Alaska Crash Data Analysis
        </h1>
        <p className="text-xs text-neutral-darker mb-4 text-center">
          Use your Django username and password (same as you use with curl or
          the admin for Basic Auth).
        </p>
        <label className="block text-xs text-neutral-darker mb-2">
          Username
          <input
            className="mt-1 w-full border border-neutral-medium rounded px-2 py-1 text-sm"
            value={auth.username}
            onChange={(e) => onFieldChange("username", e.target.value)}
          />
        </label>
        <label className="block text-xs text-neutral-darker mb-2">
          Password
          <input
            type="password"
            className="mt-1 w-full border border-neutral-medium rounded px-2 py-1 text-sm"
            value={auth.password}
            onChange={(e) => onFieldChange("password", e.target.value)}
          />
        </label>
        {auth.error && (
          <p className="text-xs text-red-600 mb-2 whitespace-pre-line text-center">
            {auth.error}
          </p>
        )}
        <button
          type="button"
          className="w-full bg-brand-primary text-white text-sm font-semibold py-2 rounded hover:bg-brand-secondary"
          onClick={onLogin}
        >
          Sign in
        </button>
      </div>
    </div>
  );
};

// ---------- Main App ----------

const App: React.FC = () => {
  // ========= DEBUG MARKER =========
  useEffect(() => {
    if (APP_VERSION_LOG_ENABLED) {
      (window as any).__APP_VERSION__ = "server-wired-v2";
      console.log("App version: server-wired-v2");
    }
  }, []);

  // File + local processing
  const [file, setFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState("");
  const [isValidating, setIsValidating] = useState(false);
  const [isPreparing, setIsPreparing] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [validationResults, setValidationResults] =
    useState<ValidationResults | null>(null);
  const [analysisResults, setAnalysisResults] =
    useState<AnalysisResults | null>(null);
  const [parsedData, setParsedData] =
    useState<Record<string, string>[] | null>(null);
  const [cleanedData, setCleanedData] =
    useState<Record<string, string>[] | null>(null);
  const [activeTab, setActiveTab] = useState<string>("Map");
  const [openDataSection, setOpenDataSection] = useState<
    "validationChecks" | "columnPlan" | "columnDetails" | null
  >(null);
  const [dataPrep, setDataPrep] = useState<DataPrepState>({
    unknownThreshold: 10,
    yesNoThreshold: 1,
    speedLimit: 70,
    roadSurface: {
      dry: true,
      wet: true,
      iceSnow: true,
    },
    leakageColumnsToDrop: [],
    columnsToDrop: [],
    severityColumn: null,
    additionalUnknownTokens: [],
  });

  const [selectedModelName, setSelectedModelName] = useState<string>(
    "crash_severity_risk_v1"
  );

  // Backend linkage: upload + model job IDs
  const [uploadId, setUploadId] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);

  // Auth
  const [auth, setAuth] = useState<AuthState>({
    username: "",
    password: "",
    isLoggedIn: false,
    error: null,
  });
  const [showAuthDropdown, setShowAuthDropdown] = useState(false);

  // Validation pipeline stage
  const [validationStage, setValidationStage] =
    useState<ValidationStage>("idle");

  // Build a Basic Auth header
  const buildAuthHeader = useCallback(() => {
    if (!auth.username || !auth.password) return {};
    const encoded = btoa(`${auth.username}:${auth.password}`);
    return { Authorization: `Basic ${encoded}` };
  }, [auth.username, auth.password]);

  // ========= AUTH / LOGIN =========

  const handleAuthFieldChange = useCallback(
    (key: "username" | "password", value: string) => {
      setAuth((prev) => ({ ...prev, [key]: value }));
    },
    []
  );

  const handleLogin = useCallback(async () => {
    if (!auth.username || !auth.password) {
      setAuth((prev) => ({
        ...prev,
        error: "Username and password are required.",
      }));
      return;
    }

    try {
      const resp = await fetch("/api/auth/login/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: auth.username,
          password: auth.password,
        }),
      });

      if (!resp.ok) {
        let msg = `Login failed (${resp.status})`;
        try {
          const data = await resp.json();
          if (data && (data.detail || data.error)) {
            msg = String(data.detail || data.error);
          }
        } catch {
          // ignore JSON parse errors
        }
        setAuth((prev) => ({ ...prev, isLoggedIn: false, error: msg }));
        return;
      }

      // Successful auth; we don't currently need the body payload here.
      setAuth((prev) => ({ ...prev, isLoggedIn: true, error: null }));
      setShowAuthDropdown(false);
    } catch (err) {
      console.error("[UI] Login error:", err);
      setAuth((prev) => ({
        ...prev,
        isLoggedIn: false,
        error:
          "Could not reach server. Is Django running on http://127.0.0.1:8000/?",
      }));
    }
  }, [auth.username, auth.password]);

  // ========= FILE SELECTION =========

  const handleFileSelect = useCallback((selectedFile: File | null) => {
    setFile(selectedFile);
    setFileName(selectedFile?.name || "");
    setValidationResults(null);
    setAnalysisResults(null);
    setParsedData(null);
    setCleanedData(null);
    setUploadId(null);
    setJobId(null);
    setValidationStage("idle");
    // Reset leakage selections for the newly chosen file
    setDataPrep((prev) => ({
      ...prev,
      leakageColumnsToDrop: [],
      columnsToDrop: [],
      severityColumn: null,
      additionalUnknownTokens: [],
    }));
  }, []);

  const handleDataPrepChange = useCallback((change: Partial<DataPrepState>) => {
    setDataPrep((prevState) => {
      const updatedState: DataPrepState = { ...prevState, ...change };
      if (change.roadSurface) {
        updatedState.roadSurface = {
          ...prevState.roadSurface,
          ...change.roadSurface,
        };
      }
      return updatedState;
    });
  }, []);

  // ========= LOCAL PARSING HELPERS =========

  const parseCsv = (fileToParse: File): Promise<Record<string, string>[]> => {
    return new Promise((resolve, reject) => {
      Papa.parse(fileToParse, {
        header: true,
        skipEmptyLines: true,
        complete: (results: { data: Record<string, string>[] }) =>
          resolve(results.data),
        error: (error: Error) => reject(error),
      });
    });
  };

  const parseXlsx = (fileToParse: File): Promise<Record<string, any>[]> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          const data = event.target?.result;
          const workbook = XLSX.read(data, { type: "array" });
          const sheetName = workbook.SheetNames[0];
          const worksheet = workbook.Sheets[sheetName];
          const json = XLSX.utils.sheet_to_json(worksheet);
          resolve(json);
        } catch (e) {
          reject(e);
        }
      };
      reader.onerror = (error) => reject(error);
      reader.readAsArrayBuffer(fileToParse);
    });
  };

  // ========= BACKEND HELPERS =========

  // Upload raw file to the ingestion gateway (/api/ingest/upload/)
  const uploadFileToBackend = useCallback(
    async (fileToUpload: File) => {
      console.log("[UI] Uploading file to /api/ingest/upload/…");
      const formData = new FormData();
      formData.append("file", fileToUpload);

      const resp = await fetch("/api/ingest/upload/", {
        method: "POST",
        headers: {
          ...buildAuthHeader(),
        },
        body: formData,
      });

      console.log("[UI] /api/ingest/upload/ status:", resp.status);

      if (!resp.ok) {
        let message = `Upload failed (${resp.status})`;
        try {
          const data = await resp.json();
          if (data && (data.detail || data.error)) {
            message = String(data.detail || data.error);
          }
        } catch {
          // ignore JSON parse / body errors
        }
        throw new Error(message);
      }

      let data: any;
      try {
        data = await resp.json();
      } catch (parseErr) {
        console.error(
          "[UI] Failed to parse JSON from /api/ingest/upload/:",
          parseErr
        );
        throw new Error(
          "Upload succeeded, but the server returned malformed JSON. Check the Django logs."
        );
      }

      if (data.upload_id) {
        setUploadId(String(data.upload_id));
      }
      return data;
    },
    [buildAuthHeader]
  );

  // Start ML job
  const startModelJob = useCallback(
    async (datasetId: string) => {
      console.log("[UI] Starting model job for dataset:", datasetId);

      // Build cleaning payload expected by analysis.ml_core.models._ensure_cleaning_params
      const cleaning: Record<string, unknown> = {
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
        const baseUnknowns = new Set<string>(BASE_UNKNOWN_TOKENS);
        dataPrep.additionalUnknownTokens.forEach((tok) => {
          const norm = tok.trim().toLowerCase();
          if (norm) {
            baseUnknowns.add(norm);
          }
        });
        cleaning.base_unknowns = Array.from(baseUnknowns);
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

      console.log("[UI] /api/models/run/ status:", resp.status);

      if (!resp.ok) {
        let msg = `Model run failed (${resp.status})`;
        try {
          const data = await resp.json();
          if (data && (data.detail || data.error)) {
            msg = String(data.detail || data.error);
          }
        } catch {
          // ignore
        }
        throw new Error(msg);
      }

      const data = await resp.json();
      if (data.job_id) setJobId(String(data.job_id));
      return data as { job_id: string; results_url?: string; status: string };
    },
    [
      buildAuthHeader,
      selectedModelName,
      dataPrep.leakageColumnsToDrop,
      dataPrep.unknownThreshold,
      dataPrep.yesNoThreshold,
      dataPrep.columnsToDrop,
      dataPrep.severityColumn,
      dataPrep.additionalUnknownTokens,
    ]
  );

  // Fetch job results
  const fetchModelResults = useCallback(
    async (modelJobId: string) => {
      console.log("[UI] Fetching results for job:", modelJobId);
      const resp = await fetch(`/api/models/results/${modelJobId}/`, {
        method: "GET",
        headers: {
          ...buildAuthHeader(),
        },
      });
      console.log("[UI] /api/models/results status:", resp.status);
      if (!resp.ok) {
        let msg = `Fetching results failed (${resp.status})`;
        try {
          const data = await resp.json();
          if (data && (data.detail || data.error)) {
            msg = String(data.detail || data.error);
          }
        } catch {
          // ignore
        }
        throw new Error(msg);
      }
      return resp.json();
    },
    [buildAuthHeader]
  );

  // ========= STEP 1: VALIDATION =========

  const handleRunValidation = useCallback(
    async () => {
      if (!file) return;

      if (!auth.isLoggedIn) {
        alert("Please sign in before running validation.");
        return;
      }

      console.log("[UI] Running validation for file:", file.name);

      setIsValidating(true);
      setValidationStage("parsing");
      setValidationResults(null);
      setAnalysisResults(null);
      setParsedData(null);
      setCleanedData(null);
      setUploadId(null);
      setJobId(null);
      setActiveTab("Data Tables");

      await new Promise((resolve) => setTimeout(resolve, 0));

      try {
        let data: Record<string, any>[];
        const lowerName = file.name.toLowerCase();

        if (lowerName.endsWith(".csv")) {
          data = await parseCsv(file);
        } else if (
          lowerName.endsWith(".xlsx") ||
          lowerName.endsWith(".xls")
        ) {
          data = await parseXlsx(file);
        } else {
          throw new Error(
            "Unsupported file type. Please upload a CSV or XLSX file."
          );
        }

        if (!data.length) {
          throw new Error("File is empty or could not be parsed correctly.");
        }

        const stringData = data.map((row) =>
          Object.fromEntries(
            Object.entries(row).map(([k, v]) => [k, String(v ?? "")])
          )
        );

        setParsedData(stringData);

        // ------------------------------------------------------------
        // Peyton-style interactive leakage selection
        // ------------------------------------------------------------
        const columns =
          stringData.length > 0 ? Object.keys(stringData[0]) : [];
        const nameSuggestions = suggestLeakageColumnsByName(columns);

        const messageLines: string[] = [];

        messageLines.push("Potential data-leakage columns.");
        messageLines.push("");
        messageLines.push(
          "These are columns that directly reveal the crash outcome, such as final injury/fatality counts or another severity code."
        );
        messageLines.push(
          "Those fields are usually NOT known at prediction time. If we let the model use them, its accuracy can look unrealistically good because it is effectively cheating with the answer."
        );
        messageLines.push("");
        messageLines.push("All feature columns:");
        columns.forEach((c) => messageLines.push("  - " + c));
        messageLines.push("");

        if (nameSuggestions.length > 0) {
          messageLines.push("Columns suspected by NAME (may indicate leakage):");
          nameSuggestions.forEach((c) => messageLines.push("  * " + c));
        } else {
          messageLines.push("No obvious leakage columns found by name keywords.");
        }

        messageLines.push("");
        messageLines.push(
          "In the next prompt you can enter a comma-separated list of columns to DROP as leakage."
        );
        messageLines.push(
          "Example: Number of Injuries, Number of Fatalities, Form Type"
        );
        messageLines.push(
          "You can include any columns from the suggestions above or others you know would reveal the outcome."
        );
        messageLines.push(
          "Press OK to continue to the input prompt, or Cancel if you want to stop and review the file instead."
        );

        const proceed = window.confirm(messageLines.join("\n"));
        if (!proceed) {
          setValidationStage("idle");
          setIsValidating(false);
          return;
        }

        const defaultValue =
          dataPrep.leakageColumnsToDrop.length > 0
            ? dataPrep.leakageColumnsToDrop.join(", ")
            : nameSuggestions.join(", ");

        const userInput = window.prompt(
          "Leakage columns to DROP (comma-separated). Example: Number of Injuries, Number of Fatalities, Form Type",
          defaultValue
        );

        let leakageColumnsToDrop = dataPrep.leakageColumnsToDrop;

        if (userInput !== null) {
          const parts = userInput
            .split(",")
            .map((p) => p.trim())
            .filter((p) => p.length > 0);

          const invalid = parts.filter((p) => !columns.includes(p));
          if (invalid.length > 0) {
            window.alert(
              "These names are not valid column names and will be ignored:\n" +
                invalid.map((p) => "  - " + p).join("\n")
            );
          }

          leakageColumnsToDrop = parts.filter((p) => columns.includes(p));
        }

        setDataPrep((prev) => ({
          ...prev,
          leakageColumnsToDrop,
        }));

        // Client-side validation / profiling
        setValidationStage("local_validation");

        let localResults = runValidationLogic(stringData, dataPrep);
        localResults = applyColumnOverrides(localResults, {
          leakageColumns: leakageColumnsToDrop,
          manualDropColumns: dataPrep.columnsToDrop,
        });
        setValidationResults(localResults);

        console.log(
          "[UI] Running client-side profiling based on Peyton’s rules…"
        );
        let richerResults = runClientValidation(stringData, dataPrep);
        richerResults = applyColumnOverrides(richerResults, {
          leakageColumns: leakageColumnsToDrop,
          manualDropColumns: dataPrep.columnsToDrop,
        });
        setValidationResults(richerResults);
        setOpenDataSection("validationChecks");

        // Upload to backend so it's stored and Peyton-cleaned in Django
        setValidationStage("uploading");
        try {
          const backendSummary = await uploadFileToBackend(file);
          console.log("[UI] Backend upload summary:", backendSummary);

          // Merge ingestion-gateway summary fields (if present) into the existing validationResults
          // so the UI can render the upload safety banner and ingestion log.
          const ingestionSummary = extractIngestionSummary(backendSummary);
          setValidationResults((prev) => ({
            ...(prev || richerResults),
            ...ingestionSummary,
          }));

          setValidationStage("complete");
        } catch (inner) {
          console.error("[UI] Backend upload failed:", inner);
          setValidationStage("error");
          const msg =
            inner instanceof Error
              ? inner.message
              : "Upload to backend failed.";
          setValidationResults((prev) => ({
            ...(prev || {
              rowCount: stringData.length,
              columnCount: stringData.length
                ? Object.keys(stringData[0]).length
                : 0,
              droppedColumnCount: 0,
              columnStats: [],
            }),
            error: msg,
          }));
        }
      } catch (err) {
        console.error("[UI] Validation error:", err);
        setValidationStage("error");
        const message =
          err instanceof Error
            ? err.message
            : "An unknown error occurred during validation.";
        setValidationResults({
          rowCount: 0,
          columnCount: 0,
          droppedColumnCount: 0,
          columnStats: [],
          error: message,
        });
        setOpenDataSection("validationChecks");
      } finally {
        setIsValidating(false);
      }
    },
    [file, dataPrep, auth.isLoggedIn, uploadFileToBackend, setDataPrep]
  );

  // ========= STEP 2: DATA PREP =========

  const handleRunDataPrep = useCallback(
    async () => {
      if (!parsedData || !validationResults) return;

      setIsPreparing(true);
      setCleanedData(null);

      await new Promise((resolve) => setTimeout(resolve, 0));

      try {
        const baseResults = runValidationLogic(parsedData, dataPrep);

        // Apply the same leakage columns the user selected during validation.
        const leakageColumns = dataPrep.leakageColumnsToDrop || [];
        const resultsWithLeakage = applyColumnOverrides(baseResults, {
          leakageColumns,
          manualDropColumns: dataPrep.columnsToDrop,
        });
        setValidationResults((prev) => ({
          ...resultsWithLeakage,
          // Preserve ingestion-gateway info captured during upload so it stays visible
          // through data prep and model selection.
          ingestionOverallStatus: prev?.ingestionOverallStatus,
          ingestionMessage: prev?.ingestionMessage,
          ingestionErrorCode: prev?.ingestionErrorCode,
          ingestionRowChecks: prev?.ingestionRowChecks,
          ingestionSteps: prev?.ingestionSteps,
        }));

        const columnsToKeep = new Set(
          resultsWithLeakage.columnStats
            .filter((c) => c.status === "Keep")
            .map((c) => c.column)
        );

        const finalCleaned = parsedData.map((row) => {
          const newRow: Record<string, string> = {};
          columnsToKeep.forEach((col) => {
            if (row[col] !== undefined) {
              newRow[col] = row[col];
            }
          });
          return newRow;
        });

        setCleanedData(finalCleaned);
        setOpenDataSection("columnDetails");
      } catch (err) {
        console.error("[UI] Data prep error:", err);
        const message =
          err instanceof Error
            ? err.message
            : "An unknown error occurred during data preparation.";
        setValidationResults({
          ...(validationResults || {
            rowCount: 0,
            columnCount: 0,
            droppedColumnCount: 0,
            columnStats: [],
          }),
          error: message,
        });
      } finally {
        setIsPreparing(false);
      }
    },
    [parsedData, dataPrep, validationResults]
  );

  // ========= STEP 3: ANALYSIS (BACKEND) =========

  const handleRunAnalysis = useCallback(
    async () => {
      if (!uploadId) {
        alert(
          "Before running analysis, please import the dataset so the backend has a stored copy (step 1)."
        );
        return;
      }

      if (!auth.isLoggedIn) {
        alert("Please sign in before running analysis.");
        return;
      }

      console.log("[UI] Running analysis for upload:", uploadId);

      setIsAnalyzing(true);
      setAnalysisResults(null);
      setActiveTab("Report Charts");

      await new Promise((resolve) => setTimeout(resolve, 0));

      try {
        const runInfo = await startModelJob(uploadId);
        const currentJobId = runInfo.job_id;
        console.log("[UI] Started model job:", currentJobId);
        setJobId(currentJobId);

        const maxAttempts = 1200;
        const delayMs = 1000;

        for (let attempt = 0; attempt < maxAttempts; attempt++) {
          const results = await fetchModelResults(currentJobId);

          if (results.status === "succeeded" && results.result_metadata) {
            const meta = results.result_metadata as any;
            const top =
              (meta.feature_importances?.top_n || []) as {
                feature: string;
                importance: number;
              }[];

            const featureImportance = top.map((item) => ({
              feature: String(item.feature),
              importance: Number(item.importance),
            }));

            const metrics = meta.metrics || {};
            const trainMetrics = metrics.train || {};
            const testMetrics = metrics.test || {};

            const reportSource =
              testMetrics.classification_report ||
              trainMetrics.classification_report ||
              metrics.test_classification_report ||
              metrics.train_classification_report ||
              {};

            const classLabels = Object.keys(reportSource).filter(
              (key) =>
                !["accuracy", "macro avg", "weighted avg"].includes(key)
            );

            const classificationReport: ClassificationReportRow[] =
              classLabels.map((label) => {
                const row = reportSource[label] || {};
                return {
                  className: label,
                  precision: Number(row.precision ?? 0),
                  recall: Number(row.recall ?? 0),
                  "f1-score": Number(row["f1-score"] ?? 0),
                  support: Number(row.support ?? 0),
                };
              });

            const leakageWarnings: string[] =
              (meta.leakage_warnings?.suspicious_features as string[]) || [];

            const decisionRules: string[] = [];

            if (featureImportance.length) {
              decisionRules.push(
                `The model's most influential feature is "${featureImportance[0].feature}". If this column looks like it directly encodes outcomes (e.g., final severity codes), consider dropping it and retraining to avoid leakage.`
              );
            }
            if (leakageWarnings.length) {
              decisionRules.push(
                `Potential leakage detected in: ${leakageWarnings.join(
                  ", "
                )}. These were flagged by Peyton's leakage detector running in the backend.`
              );
            }
            if (!decisionRules.length) {
              decisionRules.push(
                "No obvious leakage was detected. Review the top feature importances to confirm they all make sense as predictors available at prediction time."
              );
            }

            setAnalysisResults({
              featureImportance,
              decisionRules,
              classificationReport:
                classificationReport.length ? classificationReport : undefined,
              confusionMatrix: undefined,
              classLabels: classLabels.length ? classLabels : undefined,
            });
            return;
          }

          if (results.status === "failed") {
            const message =
              results.error ||
              results.result_metadata?.error ||
              "Model job failed. Check the Django logs for details.";
            setAnalysisResults({
              featureImportance: [],
              decisionRules: [],
              error: String(message),
            });
            return;
          }

          await new Promise((resolve) => setTimeout(resolve, delayMs));
        }

        setAnalysisResults({
          featureImportance: [],
          decisionRules: [],
          error:
            "Timed out waiting for model results. The job may still be running – check the Django server logs.",
        });
      } catch (err) {
        console.error("[UI] Analysis error:", err);
        const message =
          err instanceof Error
            ? err.message
            : "An unknown error occurred during analysis.";
        setAnalysisResults({
          featureImportance: [],
          decisionRules: [],
          error: message,
        });
      } finally {
        setIsAnalyzing(false);
      }
    },
    [uploadId, auth.isLoggedIn, startModelJob, fetchModelResults]
  );

  // ========= LOGIN-GATED RENDER =========

  if (!auth.isLoggedIn) {
    return (
      <LoginScreen
        auth={auth}
        onFieldChange={handleAuthFieldChange}
        onLogin={handleLogin}
      />
    );
  }

  // ========= RENDER MAIN APP =========

  return (
    <div className="min-h-screen bg-neutral-light text-neutral-dark font-sans">
      {/* HEADER WITH LOGIN DROPDOWN */}
      <header className="bg-white shadow-md p-4 flex justify-between items-center relative">
        <h1 className="text-2xl font-bold text-brand-primary">
          Alaska Crash Data Analysis Tool
        </h1>

        <div className="relative">
          <button
            type="button"
            className="flex items-center gap-2 rounded-full p-2 hover:bg-neutral-medium focus:outline-none focus:ring-2 focus:ring-brand-primary"
            onClick={() => setShowAuthDropdown((prev) => !prev)}
            aria-label={
              auth.isLoggedIn ? `Logged in as ${auth.username}` : "Sign in"
            }
          >
            <UserCircleIcon />
            <span className="hidden sm:inline text-sm text-neutral-darker">
              {auth.isLoggedIn ? auth.username : "Sign in"}
            </span>
          </button>

          {showAuthDropdown && (
            <div className="absolute right-0 mt-2 w-72 bg-white border border-neutral-medium rounded shadow-lg p-4 z-50">
              <h2 className="text-sm font-semibold mb-2 text-neutral-darker">
                {auth.isLoggedIn ? "Account" : "Connect to server"}
              </h2>

              {auth.isLoggedIn ? (
                <>
                  <p className="text-xs text-neutral-darker mb-3">
                    Signed in as{" "}
                    <span className="font-semibold">{auth.username}</span>. All
                    API requests from this page will use Basic Auth with these
                    credentials.
                  </p>
                  <button
                    type="button"
                    className="w-full border border-neutral-medium text-xs py-1.5 rounded hover:bg-neutral-light"
                    onClick={() =>
                      setAuth({
                        username: "",
                        password: "",
                        isLoggedIn: false,
                        error: null,
                      })
                    }
                  >
                    Sign out
                  </button>
                </>
              ) : (
                <>
                  <label className="block text-xs text-neutral-darker mb-1">
                    Username
                    <input
                      className="mt-1 w-full border border-neutral-medium rounded px-2 py-1 text-sm"
                      value={auth.username}
                      onChange={(e) =>
                        handleAuthFieldChange("username", e.target.value)
                      }
                    />
                  </label>
                  <label className="block text-xs text-neutral-darker mb-2">
                    Password
                    <input
                      type="password"
                      className="mt-1 w-full border border-neutral-medium rounded px-2 py-1 text-sm"
                      value={auth.password}
                      onChange={(e) =>
                        handleAuthFieldChange("password", e.target.value)
                      }
                    />
                  </label>
                  {auth.error && (
                    <p className="text-xs text-red-600 mb-2 whitespace-pre-line">
                      {auth.error}
                    </p>
                  )}
                  <button
                    type="button"
                    className="w-full bg-brand-primary text-white text-sm font-semibold py-1.5 rounded hover:bg-brand-secondary"
                    onClick={handleLogin}
                  >
                    Sign in
                  </button>
                </>
              )}
            </div>
          )}
        </div>
      </header>

      <main className="p-4 sm:p-6 lg:p-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          <div className="lg:col-span-4 xl:col-span-3">
            <WorkflowPanel
              isDatasetLoaded={!!file}
              onFileSelect={handleFileSelect}
              fileName={fileName}
              onValidationRun={handleRunValidation}
              dataPrepState={dataPrep}
              onDataPrepChange={handleDataPrepChange}
              isValidating={isValidating}
              validationStage={validationStage}
              validationResults={validationResults}
              isPreparing={isPreparing}
              onDataPrepRun={handleRunDataPrep}
              isAnalysisReady={!!cleanedData && !!uploadId}
              isAnalyzing={isAnalyzing}
              onAnalysisRun={handleRunAnalysis}
              analysisResults={analysisResults}
              selectedModelName={selectedModelName}
              onModelChange={setSelectedModelName}
            />
          </div>

          <div className="lg:col-span-8 xl:col-span-9">
            <MainContent
              activeTab={activeTab}
              setActiveTab={setActiveTab}
              uploadId={uploadId}
              authHeader={buildAuthHeader}
              isValidating={isValidating}
              validationStage={validationStage}
              validationResults={validationResults}
              isAnalyzing={isAnalyzing}
              analysisResults={analysisResults}
              openDataSection={openDataSection}
              setOpenDataSection={setOpenDataSection}
            />
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;

import { describe, expect, it } from "vitest";
import { extractIngestionSummary } from "./App";

describe("extractIngestionSummary", () => {
  it("maps ingestion gateway payload fields into ValidationResults ingestion props", () => {
    const payload = {
      upload_id: "123",
      overall_status: "accepted",
      message: "Ingestion gateway: all checks passed.",
      error_code: "NONE",
      row_checks: {
        total_rows: 1000,
        invalid_row_count: 5,
        invalid_geo_row_count: 2,
      },
      steps: [
        {
          step: "EXTENSION_CHECK",
          status: "passed",
          details: "Extension '.csv' is allowed.",
          code: "EXTENSION_ALLOWED",
        },
        {
          step: "SCHEMA_CHECK",
          status: "failed",
          severity: "error",
          details: "Missing required columns: foo, bar.",
          code: "SCHEMA_MISSING_COLUMNS",
          is_hard_fail: true,
        },
      ],
    };

    const result = extractIngestionSummary(payload);

    expect(result.ingestionOverallStatus).toBe("accepted");
    expect(result.ingestionMessage).toBe(payload.message);
    expect(result.ingestionErrorCode).toBe(payload.error_code);
    expect(result.ingestionRowChecks).toEqual(payload.row_checks);
    expect(result.ingestionSteps).toEqual([
      {
        step: "EXTENSION_CHECK",
        status: "passed",
        severity: "info",
        message: "Extension '.csv' is allowed.",
        code: "EXTENSION_ALLOWED",
        is_hard_fail: false,
      },
      {
        step: "SCHEMA_CHECK",
        status: "failed",
        severity: "error",
        message: "Missing required columns: foo, bar.",
        code: "SCHEMA_MISSING_COLUMNS",
        is_hard_fail: true,
      },
    ]);
  });

  it("supports alias fields and defaults to info severity on passed steps", () => {
    const payload = {
      overallStatus: "pending",
      detail: "Still processing.",
      steps: [
        {
          step: "ANTIVIRUS_SCAN",
          status: "passed",
          message: "File scanned successfully.",
        },
      ],
    };

    const result = extractIngestionSummary(payload);

    expect(result.ingestionOverallStatus).toBe("pending");
    expect(result.ingestionMessage).toBe("Still processing.");
    expect(result.ingestionErrorCode).toBeUndefined();
    expect(result.ingestionRowChecks).toBeUndefined();
    expect(result.ingestionSteps).toEqual([
      {
        step: "ANTIVIRUS_SCAN",
        status: "passed",
        severity: "info",
        message: "File scanned successfully.",
        is_hard_fail: false,
      },
    ]);
  });
});

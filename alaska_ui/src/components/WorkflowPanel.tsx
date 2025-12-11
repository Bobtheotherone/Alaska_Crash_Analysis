import React, { useState, useEffect } from "react";
import CollapsibleSection from "./CollapsibleSection";
import {
  DataPrepState,
  ValidationResults,
  AnalysisResults,
  ValidationStage,
} from "../App";

interface WorkflowPanelProps {
  isDatasetLoaded: boolean;
  fileName: string;
  onFileSelect: (file: File | null) => void;
  onValidationRun: () => void;
  dataPrepState: DataPrepState;
  onDataPrepChange: (change: Partial<DataPrepState>) => void;
  isValidating: boolean;
  validationStage: ValidationStage;
  validationResults: ValidationResults | null;
  isPreparing: boolean;
  onDataPrepRun: () => void;
  isAnalysisReady: boolean;
  isAnalyzing: boolean;
  onAnalysisRun: () => void;
  analysisResults: AnalysisResults | null;
  selectedModelName: string;
  onModelChange: (modelName: string) => void;
}

const MODEL_OPTIONS = [
  { value: "crash_severity_risk_v1", label: "Decision Tree" },
  { value: "mrf_v1", label: "Multilevel Random Forest" },
  { value: "xgb_v1", label: "XGBoost" },
  { value: "ebm_v1", label: "Explainable Boosting Machine (EBM)" },
];

const WorkflowPanel: React.FC<WorkflowPanelProps> = ({
  isDatasetLoaded,
  fileName,
  onFileSelect,
  onValidationRun,
  dataPrepState,
  onDataPrepChange,
  isValidating,
  validationStage,
  validationResults,
  isPreparing,
  onDataPrepRun,
  isAnalysisReady,
  isAnalyzing,
  onAnalysisRun,
  analysisResults,
  selectedModelName,
  onModelChange,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [activeSection, setActiveSection] = useState<
    "import" | "prep" | "analysis" | null
  >("import");

  useEffect(() => {
    if (validationResults && !validationResults.error) {
      setActiveSection("prep");
    }
  }, [validationResults]);

  useEffect(() => {
    if (isAnalysisReady) {
      setActiveSection("analysis");
    }
  }, [isAnalysisReady]);

  const handleToggle = (section: "import" | "prep" | "analysis") => {
    setActiveSection((prev) => (prev === section ? null : section));
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      onFileSelect(event.target.files[0]);
    } else {
      onFileSelect(null);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileSelect(e.dataTransfer.files[0]);
      e.dataTransfer.clearData();
    }
  };

  const getStatus = (
    step: "import" | "prep" | "analysis"
  ): { type: "ready" | "valid" | "error"; text: string } | null => {
    const truncateError = (err: string | undefined) => {
      if (!err) return "An unknown error occurred.";
      return err.length > 35 ? err.substring(0, 35) + "..." : err;
    };

    switch (step) {
      case "import": {
        if (validationResults?.error) {
          return {
            type: "error",
            text: truncateError(validationResults.error),
          };
        }

        switch (validationStage) {
          case "parsing":
            return { type: "ready", text: "Cleaning & parsing data…" };
          case "local_validation":
            return { type: "ready", text: "Validating dataset…" };
          case "uploading":
            return { type: "ready", text: "Saving dataset to server…" };
          case "complete":
            if (validationResults && !validationResults.error) {
              return { type: "valid", text: "Schema valid" };
            }
            return { type: "ready", text: "Ready" };
          case "error":
            return {
              type: "error",
              text: truncateError(validationResults?.error),
            };
          case "idle":
          default:
            if (isDatasetLoaded && !isValidating) {
              return { type: "ready", text: "Ready" };
            }
            return null;
        }
      }

      case "prep": {
        if (isPreparing) {
          return { type: "ready", text: "Cleaning & selecting features…" };
        }
        if (isAnalysisReady) {
          return { type: "valid", text: "Done" };
        }
        if (validationResults && !validationResults.error) {
          return { type: "ready", text: "Ready" };
        }
        return null;
      }

      case "analysis": {
        if (analysisResults?.error) {
          return {
            type: "error",
            text: truncateError(analysisResults.error),
          };
        }
        if (isAnalyzing) {
          return { type: "ready", text: "Running model…" };
        }
        if (analysisResults) {
          return { type: "valid", text: "Done" };
        }
        if (isAnalysisReady) {
          return { type: "ready", text: "Ready" };
        }
        return null;
      }

      default:
        return null;
    }
  };

  const importStatus = getStatus("import");
  const dataPrepStatus = getStatus("prep");
  const analysisStatus = getStatus("analysis");

  const validationButtonLabel =
    validationStage === "parsing"
      ? "Cleaning data…"
      : validationStage === "local_validation"
      ? "Validating dataset…"
      : validationStage === "uploading"
      ? "Saving to server…"
      : isValidating
      ? "Validating…"
      : "Run Validator";

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 space-y-6">
      <h2 className="text-xl font-semibold text-neutral-darker border-b border-neutral-medium pb-4">
        Analysis Workflow
      </h2>

      {/* 1. Import / Validate */}
      <CollapsibleSection
        title="1. Import Dataset"
        isOpen={activeSection === "import"}
        onToggle={() => handleToggle("import")}
        status={importStatus}
      >
        <div className="space-y-4">
          <label
            htmlFor="file-upload"
            className={`w-full text-center px-4 py-6 border-2 border-dashed rounded-lg cursor-pointer transition-colors duration-300 flex flex-col items-center justify-center ${
              isDragging
                ? "border-brand-primary bg-blue-100"
                : "border-neutral-medium hover:border-brand-secondary hover:bg-blue-50"
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <svg
              className="w-8 h-8 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M7 16a4 4 0 01-4-4V6a4 4 0 014-4h10a4 4 0 014 4v6a4 4 0 01-4 4H7z"
              ></path>
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M16 16v1a2 2 0 01-2 2H6a2 2 0 01-2-2v-1"
              ></path>
            </svg>
            <span className="mt-2 text-sm text-gray-600 truncate">
              {fileName || "Click to upload or drag and drop"}
            </span>
            <span className="text-xs text-gray-500">CSV, XLSX, Parquet</span>
          </label>
          <input
            id="file-upload"
            type="file"
            className="hidden"
            accept=".csv,.xlsx,.xls,.parquet"
            onChange={handleFileChange}
          />
          <button
            onClick={onValidationRun}
            disabled={!isDatasetLoaded || isValidating}
            className="w-full bg-brand-primary text-white font-bold py-2 px-4 rounded-lg transition-colors duration-300 hover:bg-brand-secondary disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center"
          >
            {isValidating && (
              <svg
                className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
            )}
            {validationButtonLabel}
          </button>
        </div>
      </CollapsibleSection>

      {/* 2. Data Preparation */}
      <CollapsibleSection
        title="2. Data Preparation"
        isOpen={activeSection === "prep"}
        onToggle={() => handleToggle("prep")}
        status={dataPrepStatus}
      >
        <div className="space-y-4 text-sm">
          <fieldset
            disabled={!validationResults || !!validationResults.error}
            className="disabled:opacity-50 space-y-4"
          >
            <div>
              <div className="flex justify-between items-center">
                <label
                  htmlFor="unknown-threshold"
                  className="font-medium text-gray-700"
                >
                  Exclude data if % unknown exceeds
                </label>
                <span className="text-sm font-semibold text-brand-primary tabular-nums">
                  {dataPrepState.unknownThreshold}%
                </span>
              </div>
              <input
                type="range"
                id="unknown-threshold"
                min="0"
                max="100"
                value={dataPrepState.unknownThreshold}
                onChange={(e) =>
                  onDataPrepChange({
                    unknownThreshold: parseInt(e.target.value, 10),
                  })
                }
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer mt-1"
              />
            </div>
            <div>
              <div className="flex justify-between items-center">
                <label
                  htmlFor="yes-no-threshold"
                  className="font-medium text-gray-700"
                >
                  Yes/No Imbalance Threshold
                </label>
                <span className="text-sm font-semibold text-brand-primary tabular-nums">
                  {dataPrepState.yesNoThreshold}%
                </span>
              </div>
              <input
                type="range"
                id="yes-no-threshold"
                min="0"
                max="50"
                value={dataPrepState.yesNoThreshold}
                onChange={(e) =>
                  onDataPrepChange({
                    yesNoThreshold: parseInt(e.target.value, 10),
                  })
                }
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer mt-1"
              />
            </div>
            <div>
              <div className="flex justify-between items-center">
                <label
                  htmlFor="speed-limit"
                  className="font-medium text-gray-700"
                >
                  Max Posted Speed Limit (MPH)
                </label>
                <span className="text-sm font-semibold text-brand-primary tabular-nums">
                  {dataPrepState.speedLimit} MPH
                </span>
              </div>
              <input
                type="range"
                id="speed-limit"
                min="0"
                max="85"
                value={dataPrepState.speedLimit}
                onChange={(e) =>
                  onDataPrepChange({
                    speedLimit: parseInt(e.target.value, 10),
                  })
                }
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer mt-1"
              />
            </div>
            <div>
              <span className="block font-medium text-gray-700 mb-2">
                Road Surface
              </span>
              <div className="space-y-1">
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="surface-dry"
                    className="h-4 w-4 rounded border-gray-300 accent-brand-primary focus:ring-brand-secondary"
                    checked={dataPrepState.roadSurface.dry}
                    onChange={(e) =>
                      onDataPrepChange({
                        roadSurface: {
                          ...dataPrepState.roadSurface,
                          dry: e.target.checked,
                        },
                      })
                    }
                  />
                  <label
                    htmlFor="surface-dry"
                    className="ml-2 text-gray-600"
                  >
                    Dry
                  </label>
                </div>
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="surface-wet"
                    className="h-4 w-4 rounded border-gray-300 accent-brand-primary focus:ring-brand-secondary"
                    checked={dataPrepState.roadSurface.wet}
                    onChange={(e) =>
                      onDataPrepChange({
                        roadSurface: {
                          ...dataPrepState.roadSurface,
                          wet: e.target.checked,
                        },
                      })
                    }
                  />
                  <label
                    htmlFor="surface-wet"
                    className="ml-2 text-gray-600"
                  >
                    Wet
                  </label>
                </div>
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="surface-ice"
                    className="h-4 w-4 rounded border-gray-300 accent-brand-primary focus:ring-brand-secondary"
                    checked={dataPrepState.roadSurface.iceSnow}
                    onChange={(e) =>
                      onDataPrepChange({
                        roadSurface: {
                          ...dataPrepState.roadSurface,
                          iceSnow: e.target.checked,
                        },
                      })
                    }
                  />
                  <label
                    htmlFor="surface-ice"
                    className="ml-2 text-gray-600"
                  >
                    Ice/Snow
                  </label>
                </div>
              </div>
            </div>
          </fieldset>
          <button
            onClick={onDataPrepRun}
            disabled={
              !validationResults || !!validationResults.error || isPreparing
            }
            className="w-full bg-brand-primary text-white font-bold py-2 px-4 rounded-lg transition-colors duration-300 hover:bg-brand-secondary disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center mt-4"
          >
            {isPreparing && (
              <svg
                className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
            )}
            {isPreparing ? "Preparing Data..." : "Run Data Preparation"}
          </button>
        </div>
      </CollapsibleSection>

      {/* 3. Analysis */}
      <CollapsibleSection
        title="3. Select ML Algorithm"
        isOpen={activeSection === "analysis"}
        onToggle={() => handleToggle("analysis")}
        status={analysisStatus}
      >
        <div className="space-y-4">
          <fieldset
            disabled={!isDatasetLoaded}
            className="disabled:opacity-50 space-y-4"
          >
            <div>
              <label htmlFor="ml-algorithm" className="sr-only">
                Select ML Algorithm
              </label>
              <select
                id="ml-algorithm"
                className="w-full bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-brand-primary focus:border-brand-primary block p-2.5"
                value={selectedModelName}
                onChange={(e) => onModelChange(e.target.value)}
              >
                {MODEL_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
            <button
              onClick={onAnalysisRun}
              disabled={!isAnalysisReady || isAnalyzing}
              className="w-full bg-brand-primary text-white font-bold py-2 px-4 rounded-lg transition-colors duration-300 hover:bg-brand-secondary disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {isAnalyzing && (
                <svg
                  className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
              )}
              {isAnalyzing ? "Analyzing..." : "Run Analysis"}
            </button>
            <button className="w-full bg-gray-600 text-white font-bold py-2 px-4 rounded-lg transition-colors duration-300 hover:bg-gray-700 disabled:bg-gray-400 disabled:cursor-not-allowed">
              Export Report (PDF)
            </button>
          </fieldset>
        </div>
      </CollapsibleSection>
    </div>
  );
};

export default WorkflowPanel;

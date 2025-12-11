import React from 'react';
import { ValidationResults, AnalysisResults, ValidationStage } from './App';
import ValidationResultsDisplay from './ValidationResults';
import ReportCharts from './ReportCharts';
import ClassificationResults from './ClassificationResults';

interface MainContentProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  isValidating: boolean;
  validationStage: ValidationStage;
  validationResults: ValidationResults | null;
  openDataSection?: 'validationChecks' | 'columnPlan' | 'columnDetails' | null;
  setOpenDataSection?: (
    section: 'validationChecks' | 'columnPlan' | 'columnDetails' | null
  ) => void;
  isAnalyzing: boolean;
  analysisResults: AnalysisResults | null;
  onExportValidationCsv?: () => void;
  canExportValidationCsv?: boolean;
}

const TABS = ['Map', 'Data Tables', 'Report Charts', 'Classifications', 'EBM'];

const TabContent: React.FC<{
  activeTab: string;
  isValidating: boolean;
  validationStage: ValidationStage;
  validationResults: ValidationResults | null;
  isAnalyzing: boolean;
  analysisResults: AnalysisResults | null;
  openDataSection?: 'validationChecks' | 'columnPlan' | 'columnDetails' | null;
  setOpenDataSection?: (
    section: 'validationChecks' | 'columnPlan' | 'columnDetails' | null
  ) => void;
  onExportValidationCsv?: () => void;
  canExportValidationCsv?: boolean;
}> = ({
  activeTab,
  isValidating,
  validationStage,
  validationResults,
  isAnalyzing,
  analysisResults,
  openDataSection,
  setOpenDataSection,
  onExportValidationCsv,
  canExportValidationCsv,
}) => {
  switch (activeTab) {
    case 'Map':
      return (
        <div className="h-full w-full bg-gray-200 rounded-lg overflow-hidden">
          <iframe
            title="Map of Alaska"
            width="100%"
            height="100%"
            loading="lazy"
            allowFullScreen
            src="https://www.openstreetmap.org/export/embed.html?bbox=-179.14734,51.20978,-129.97955,71.38957&layer=mapnik"
            className="border-0"
          ></iframe>
        </div>
      );

    case 'Data Tables': {
      const isBusy =
        isValidating &&
        ['parsing', 'local_validation', 'uploading'].includes(validationStage);

      if (isBusy) {
        return (
          <div className="p-8 text-center text-gray-500 flex flex-col items-center justify-center h-full">
            <svg
              className="animate-spin h-10 w-10 text-brand-primary mb-4"
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
            <h3 className="text-xl font-semibold">Validating dataset…</h3>
            <p>Please wait while we analyze your data and save it to the ingestion gateway.</p>
          </div>
        );
      }

      if (validationResults) {
        return (
          <div className="flex flex-col h-full">
            {/* Top bar: export button only (we intentionally omit the old
                "Validation summary" heading to match the desired layout). */}
            <div className="mb-4 flex justify-end">
              {onExportValidationCsv && (
                <div className="shrink-0">
                  <button
                    type="button"
                    onClick={onExportValidationCsv}
                    disabled={!canExportValidationCsv}
                    className="inline-flex items-center rounded-md border border-neutral-medium bg-white px-3 py-1.5 text-xs font-medium text-neutral-darker hover:bg-neutral-light disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <span className="hidden sm:inline">Export validation report (CSV)</span>
                    <span className="sm:hidden">Export CSV</span>
                  </button>
                </div>
              )}
            </div>

            <div className="flex-1 min-h-0 overflow-y-auto">
              <ValidationResultsDisplay
                results={validationResults}
                openDataSection={openDataSection}
                setOpenDataSection={setOpenDataSection}
              />
            </div>
          </div>
        );
      }

      return (
        <div className="p-8 text-center text-gray-500">
          <h3 className="text-xl font-semibold">Data Tables</h3>
          <p>Run the validator on an imported dataset to see the results here.</p>
        </div>
      );
    }

    case 'Report Charts':
      if (isAnalyzing) {
        return (
          <div className="p-8 text-center text-gray-500 flex flex-col items-center justify-center h-full">
            <svg
              className="animate-spin h-10 w-10 text-brand-primary mb-4"
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
            <h3 className="text-xl font-semibold">Running Analysis...</h3>
            <p>Please wait while we generate insights from your data.</p>
          </div>
        );
      }
      if (analysisResults) {
        return <ReportCharts results={analysisResults} />;
      }
      return (
        <div className="p-8 text-center text-gray-500">
          <h3 className="text-xl font-semibold">Report Charts</h3>
          <p>Run an analysis on a prepared dataset to see the results here.</p>
        </div>
      );

    case 'Classifications':
      if (isAnalyzing) {
        return (
          <div className="p-8 text-center text-gray-500 flex flex-col items-center justify-center h-full">
            <svg
              className="animate-spin h-10 w-10 text-brand-primary mb-4"
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
            <h3 className="text-xl font-semibold">Running Analysis...</h3>
            <p>Please wait while we generate insights from your data.</p>
          </div>
        );
      }
      if (analysisResults) {
        return <ClassificationResults results={analysisResults} />;
      }
      return (
        <div className="p-8 text-center text-gray-500">
          <h3 className="text-xl font-semibold">Classifications</h3>
          <p>Run an analysis on a prepared dataset to see the classification results here.</p>
        </div>
      );

    default:
      return (
        <div className="p-8 text-center text-gray-500">
          <h3 className="text-xl font-semibold">{activeTab}</h3>
          <p>Content for this view will be displayed here.</p>
        </div>
      );
  }
};

const MainContent: React.FC<MainContentProps> = ({
  activeTab,
  setActiveTab,
  isValidating,
  validationStage,
  validationResults,
  isAnalyzing,
  analysisResults,
  openDataSection,
  setOpenDataSection,
  onExportValidationCsv,
  canExportValidationCsv,
}) => {
  return (
    <div className="bg-white rounded-lg shadow-lg h-[80vh]">
      <div className="border-b border-neutral-medium">
        <nav className="-mb-px flex space-x-6 px-6" aria-label="Tabs">
          {TABS.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`${
                activeTab === tab
                  ? 'border-brand-primary text-brand-primary'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm transition-colors duration-200`}
            >
              {tab}
            </button>
          ))}
        </nav>
      </div>
      <div className="p-6 h-[calc(80vh-65px)]">
        <TabContent
          activeTab={activeTab}
          isValidating={isValidating}
          validationStage={validationStage}
          validationResults={validationResults}
          isAnalyzing={isAnalyzing}
          analysisResults={analysisResults}
          openDataSection={openDataSection}
          setOpenDataSection={setOpenDataSection}
          onExportValidationCsv={onExportValidationCsv}
          canExportValidationCsv={canExportValidationCsv}
        />
      </div>
    </div>
  );
};

export default MainContent;

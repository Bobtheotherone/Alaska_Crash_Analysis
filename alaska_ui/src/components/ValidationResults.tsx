import React, { useState, useMemo, useEffect } from 'react';
import { ValidationResults, ColumnStat } from './App';
import UploadSafetySummary from './UploadSafetySummary';
import ColumnPlanSummary from './ColumnPlanSummary';
import IngestionLogAccordion from './IngestionLogAccordion';

interface ValidationResultsDisplayProps {
  results: ValidationResults;
  openDataSection?: 'validationChecks' | 'columnPlan' | 'columnDetails' | null;
  setOpenDataSection?: (
    section: 'validationChecks' | 'columnPlan' | 'columnDetails' | null
  ) => void;
}

type SortKey = keyof ColumnStat | 'yesNoCoverage';
type SortDirection = 'asc' | 'desc';

const getSortValue = (col: ColumnStat, key: SortKey): number | string => {
  if (key === 'yesNoCoverage') {
    return col.yesNoStats ? col.yesNoStats.coveragePercent : -1;
  }
  if (key === 'unknownPercent') {
    return col.unknownPercent;
  }
  if (key === 'column') {
    return col.column.toLowerCase();
  }
  return (col as any)[key] ?? '';
};

const ValidationResultsDisplay: React.FC<ValidationResultsDisplayProps> = ({
  results,
  openDataSection,
  setOpenDataSection,
}) => {
  const [sortConfig, setSortConfig] = useState<{ key: SortKey; direction: SortDirection }>({
    key: 'column',
    direction: 'asc',
  });

  // Which dropdowns are open
  const [validationOpen, setValidationOpen] = useState<boolean>(true);
  const [columnDetailsOpen, setColumnDetailsOpen] = useState<boolean>(false);

  // React to hints from App.tsx about which section should be emphasized
  useEffect(() => {
    if (!openDataSection) {
      return;
    }

    if (openDataSection === 'validationChecks') {
      // After step 1: show validation/ingestion, hide column-by-column table
      setValidationOpen(true);
      setColumnDetailsOpen(false);
    } else if (openDataSection === 'columnDetails') {
      // After step 2: focus on column-by-column details, collapse validation section
      setValidationOpen(false);
      setColumnDetailsOpen(true);
    }

    // Consume the one-shot hint so it doesn't keep firing
    setOpenDataSection?.(null);
  }, [openDataSection, setOpenDataSection]);

  const sortedColumnStats = useMemo(() => {
    const sorted = [...results.columnStats];
    sorted.sort((a, b) => {
      const aVal = getSortValue(a, sortConfig.key);
      const bVal = getSortValue(b, sortConfig.key);

      if (aVal < (bVal as any)) {
        return sortConfig.direction === 'asc' ? -1 : 1;
      }
      if (aVal > (bVal as any)) {
        return sortConfig.direction === 'asc' ? 1 : -1;
      }
      return 0;
    });
    return sorted;
  }, [results.columnStats, sortConfig]);

  const handleSort = (key: SortKey) => {
    setSortConfig((prev) => {
      if (prev.key === key) {
        return {
          key,
          direction: prev.direction === 'asc' ? 'desc' : 'asc',
        };
      }
      return { key, direction: 'asc' };
    });
  };

  const renderSortIcon = (key: SortKey) => {
    if (sortConfig.key !== key) {
      return (
        <span className="inline-flex flex-col ml-1 text-gray-300">
          <span className="leading-none">▲</span>
          <span className="leading-none -mt-1">▼</span>
        </span>
      );
    }
    return (
      <span className="inline-flex flex-col ml-1 text-gray-700">
        <span
          className={`leading-none ${
            sortConfig.direction === 'asc' ? 'text-brand-primary' : ''
          }`}
        >
          ▲
        </span>
        <span
          className={`leading-none -mt-1 ${
            sortConfig.direction === 'desc' ? 'text-brand-primary' : ''
          }`}
        >
          ▼
        </span>
      </span>
    );
  };

  const ingestionRejected =
    String(results.ingestionOverallStatus || '').toLowerCase() === 'rejected';
  const hasError = !!results.error || ingestionRejected;

  return (
    <div className="bg-white rounded-lg border border-neutral-medium">
      <div className="p-4 space-y-4">
        {/* Always-visible column plan at the very top */}
        <ColumnPlanSummary results={results} />

        {/* Dropdown 1: Validation Results (ingestion + safety checks) */}
        <section className="border border-neutral-light rounded-md bg-white">
          <button
            type="button"
            className="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-gray-50"
            onClick={() => setValidationOpen((open) => !open)}
          >
            <div>
              <h3 className="text-sm font-semibold text-neutral-darker">
                Validation Results
              </h3>
              <p className="text-xs text-gray-500">
                Column-by-column summary of unknown values, Yes/No balance, and ingestion gateway
                checks.
              </p>
            </div>
            <div className="flex items-center gap-3">
              <span
                className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-[11px] font-semibold border ${
                  hasError
                    ? 'bg-red-50 text-red-700 border-red-200'
                    : 'bg-green-50 text-green-700 border-green-200'
                }`}
              >
                {hasError ? 'Needs attention' : 'Ready'}
              </span>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className={`h-4 w-4 text-gray-500 transform transition-transform duration-200 ${
                  validationOpen ? '-rotate-180' : ''
                }`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 9l-7 7-7-7"
                />
              </svg>
            </div>
          </button>

          {validationOpen && (
            <div className="border-t border-neutral-light p-3 space-y-3">
              <UploadSafetySummary results={results} />
              <IngestionLogAccordion results={results} />
            </div>
          )}
        </section>

        {/* Dropdown 2: Column-by-column details */}
        <section className="border border-neutral-light rounded-md bg-white">
          <button
            type="button"
            className="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-gray-50"
            onClick={() => setColumnDetailsOpen((open) => !open)}
          >
            <div>
              <h3 className="text-sm font-semibold text-neutral-darker">
                Column-by-column details
              </h3>
              <p className="text-xs text-gray-500">
                Explore unknown-value percentages and Yes/No coverage for each column.
              </p>
            </div>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className={`h-4 w-4 text-gray-500 transform transition-transform duration-200 ${
                columnDetailsOpen ? '-rotate-180' : ''
              }`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 9l-7 7-7-7"
              />
            </svg>
          </button>

          {columnDetailsOpen && (
            <div className="border-t border-neutral-light overflow-x-auto">
              <table className="min-w-full divide-y divide-neutral-light text-xs">
                <thead className="bg-gray-50">
                  <tr>
                    <th
                      scope="col"
                      className="px-3 py-2 text-left font-semibold text-gray-700 cursor-pointer"
                      onClick={() => handleSort('column')}
                    >
                      <span className="inline-flex items-center">
                        Column
                        {renderSortIcon('column')}
                      </span>
                    </th>
                    <th
                      scope="col"
                      className="px-3 py-2 text-right font-semibold text-gray-700 cursor-pointer"
                      onClick={() => handleSort('unknownPercent')}
                    >
                      <span className="inline-flex items-center">
                        Unknown %
                        {renderSortIcon('unknownPercent')}
                      </span>
                    </th>
                    <th
                      scope="col"
                      className="px-3 py-2 text-right font-semibold text-gray-700 cursor-pointer"
                      onClick={() => handleSort('yesNoCoverage')}
                    >
                      <span className="inline-flex items-center">
                        Yes/No coverage
                        {renderSortIcon('yesNoCoverage')}
                      </span>
                    </th>
                    <th
                      scope="col"
                      className="px-3 py-2 text-left font-semibold text-gray-700"
                    >
                      Status
                    </th>
                    <th
                      scope="col"
                      className="px-3 py-2 text-left font-semibold text-gray-700"
                    >
                      Reason
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-neutral-light">
                  {sortedColumnStats.map((col) => (
                    <tr key={col.column} className="hover:bg-gray-50">
                      <td className="px-3 py-2 whitespace-nowrap text-gray-900">
                        {col.column}
                      </td>
                      <td className="px-3 py-2 text-right text-gray-700">
                        {col.unknownPercent.toFixed(1)}%
                      </td>
                      <td className="px-3 py-2 text-right text-gray-700">
                        {col.yesNoStats ? (
                          <>
                            {col.yesNoStats.coveragePercent.toFixed(1)}% of rows
                          </>
                        ) : (
                          <span className="italic text-gray-400">N/A</span>
                        )}
                      </td>
                      <td className="px-3 py-2 text-left">
                        <span
                          className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold ${
                            col.status === 'Keep'
                              ? 'bg-green-100 text-green-800'
                              : 'bg-red-100 text-red-800'
                          }`}
                        >
                          {col.status}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-left text-gray-700">
                        {col.reason || <span className="italic text-gray-400">—</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </div>
  );
};

export default ValidationResultsDisplay;

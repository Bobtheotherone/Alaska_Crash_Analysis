import React, { useState, useMemo, useEffect } from 'react';
import { ValidationResults, ColumnStat } from './App';
import UploadSafetySummary from './UploadSafetySummary';
import ColumnPlanSummary from './ColumnPlanSummary';
import IngestionLogAccordion from './IngestionLogAccordion';

interface ValidationResultsDisplayProps {
  results: ValidationResults;
  autoExpandColumnDetails?: boolean;
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
  autoExpandColumnDetails,
}) => {
  const [sortConfig, setSortConfig] = useState<{ key: SortKey; direction: SortDirection }>({
    key: 'column',
    direction: 'asc',
  });
  const [detailsOpen, setDetailsOpen] = useState(true);

  useEffect(() => {
    if (autoExpandColumnDetails) {
      setDetailsOpen(true);
    }
  }, [autoExpandColumnDetails]);

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

  return (
    <div className="bg-white rounded-b-lg">
      <div className="p-4 space-y-4">
        <UploadSafetySummary results={results} />
        <ColumnPlanSummary results={results} />
        <IngestionLogAccordion results={results} />

        <section className="border border-neutral-light rounded-md bg-white">
          <button
            type="button"
            className="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-gray-50"
            onClick={() => setDetailsOpen((open) => !open)}
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
                detailsOpen ? '-rotate-180' : ''
              }`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {detailsOpen && (
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

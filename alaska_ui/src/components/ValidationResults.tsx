import React, { useState, useMemo } from 'react';
import { ValidationResults, ColumnStat } from '../App';
import { ChevronDownIcon } from '../constants';

interface ValidationResultsDisplayProps {
  results: ValidationResults;
}

type SortKey = keyof ColumnStat | 'yesNoCoverage';
type SortDirection = 'asc' | 'desc';

const SortableHeader: React.FC<{
  title: string;
  sortKey: SortKey;
  currentSortKey: SortKey;
  sortDirection: SortDirection;
  onSort: (key: SortKey) => void;
}> = ({ title, sortKey, currentSortKey, sortDirection, onSort }) => (
  <th
    scope="col"
    className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
    onClick={() => onSort(sortKey)}
  >
    <div className="flex items-center">
      <span>{title}</span>
      {currentSortKey === sortKey && (
        <span className="ml-1">
          {sortDirection === 'asc' ? '▲' : '▼'}
        </span>
      )}
    </div>
  </th>
);

const ValidationResultsDisplay: React.FC<ValidationResultsDisplayProps> = ({ results }) => {
  const [sortConfig, setSortConfig] = useState<{ key: SortKey; direction: SortDirection }>({ key: 'column', direction: 'asc' });

  const sortedColumnStats = useMemo(() => {
    let sortableItems = [...results.columnStats];
    sortableItems.sort((a, b) => {
      let aValue: any;
      let bValue: any;

      if (sortConfig.key === 'yesNoCoverage') {
        aValue = a.yesNoStats?.coveragePercent ?? -1;
        bValue = b.yesNoStats?.coveragePercent ?? -1;
      } else {
        aValue = a[sortConfig.key as keyof ColumnStat];
        bValue = b[sortConfig.key as keyof ColumnStat];
      }
      
      if (aValue < bValue) {
        return sortConfig.direction === 'asc' ? -1 : 1;
      }
      if (aValue > bValue) {
        return sortConfig.direction === 'asc' ? 1 : -1;
      }
      return 0;
    });
    return sortableItems;
  }, [results.columnStats, sortConfig]);

  const handleSort = (key: SortKey) => {
    let direction: SortDirection = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };
  
  if (results.error) {
    return (
      <div className="p-8 text-center text-red-700 bg-red-100 rounded-lg">
        <h3 className="text-xl font-semibold">Validation Failed</h3>
        <p>{results.error}</p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col space-y-4">
      <div className="bg-blue-50 border border-blue-200 p-4 rounded-lg grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
        <div>
          <p className="text-sm text-gray-600">Rows</p>
          <p className="text-2xl font-bold text-brand-primary">{results.rowCount.toLocaleString()}</p>
        </div>
        <div>
          <p className="text-sm text-gray-600">Columns</p>
          <p className="text-2xl font-bold text-brand-primary">{results.columnCount}</p>
        </div>
        <div>
          <p className="text-sm text-gray-600">Columns to Drop</p>
          <p className="text-2xl font-bold text-red-600">{results.droppedColumnCount}</p>
        </div>
      </div>
      <div className="flex-grow overflow-auto border border-neutral-medium rounded-lg">
        <table className="min-w-full divide-y divide-neutral-medium">
          <thead className="bg-gray-50 sticky top-0">
            <tr>
              <SortableHeader title="Column" sortKey="column" currentSortKey={sortConfig.key} sortDirection={sortConfig.direction} onSort={handleSort} />
              <SortableHeader title="% Unknown" sortKey="unknownPercent" currentSortKey={sortConfig.key} sortDirection={sortConfig.direction} onSort={handleSort} />
              <SortableHeader title="Yes/No Coverage" sortKey="yesNoCoverage" currentSortKey={sortConfig.key} sortDirection={sortConfig.direction} onSort={handleSort} />
              <SortableHeader title="Status" sortKey="status" currentSortKey={sortConfig.key} sortDirection={sortConfig.direction} onSort={handleSort} />
              <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Reason</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {sortedColumnStats.map((col) => (
              <tr key={col.column} className="hover:bg-gray-50">
                <td className="px-4 py-3 text-sm font-medium text-gray-900 max-w-xs break-words">{col.column}</td>
                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500 tabular-nums">{col.unknownPercent.toFixed(2)}%</td>
                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500 tabular-nums">
                  {col.yesNoStats ? `${col.yesNoStats.coveragePercent.toFixed(2)}%` : 'N/A'}
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-sm">
                  <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${col.status === 'Keep' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                    {col.status}
                  </span>
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">{col.reason}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ValidationResultsDisplay;
import React from 'react';
import { ValidationResults } from './App';
import StatusPill from './ui/StatusPill';

interface ColumnPlanSummaryProps {
  results: ValidationResults;
}

const ColumnPlanSummary: React.FC<ColumnPlanSummaryProps> = ({ results }) => {
  const keptColumns = results.columnStats.filter((col) => col.status === 'Keep');
  const droppedColumns = results.columnStats.filter((col) => col.status === 'Drop');

  const keptCount = keptColumns.length;
  const droppedCount = results.droppedColumnCount;

  return (
    <section className="rounded-md border border-blue-100 bg-blue-50/60 px-4 py-3 space-y-3">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div>
          <h3 className="text-sm font-semibold text-neutral-darker">Column plan</h3>
          <p className="text-xs text-gray-600">
            Dropped columns either exceed the unknown-value threshold or have highly imbalanced
            Yes/No coverage based on your current settings.
          </p>
        </div>
        <StatusPill tone="info">
          {keptCount} kept / {droppedCount} dropped
        </StatusPill>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm">
        <div className="bg-white/60 rounded-lg p-3">
          <p className="text-xs text-gray-500">Rows</p>
          <p className="text-lg font-semibold text-neutral-darker">
            {results.rowCount.toLocaleString()}
          </p>
        </div>
        <div className="bg-white/60 rounded-lg p-3">
          <p className="text-xs text-gray-500">Columns</p>
          <p className="text-lg font-semibold text-neutral-darker">
            {results.columnCount.toLocaleString()}
          </p>
        </div>
        <div className="bg-white/60 rounded-lg p-3">
          <p className="text-xs text-gray-500">Dropped columns</p>
          <p className="text-lg font-semibold text-red-600">
            {droppedCount.toLocaleString()}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 text-xs">
        <div>
          <h4 className="text-sm font-semibold text-neutral-darker mb-2 flex items-center">
            Kept columns{' '}
            <span className="ml-2 text-xs text-gray-500">({keptColumns.length})</span>
          </h4>
          <div className="space-y-1 max-h-40 overflow-y-auto border border-neutral-light rounded-md p-2 bg-white">
            {keptColumns.map((col) => (
              <div
                key={col.column}
                className="flex justify-between items-center"
              >
                <span className="font-medium text-neutral-darker">{col.column}</span>
                {col.yesNoStats && (
                  <span className="text-gray-500">
                    Yes: {col.yesNoStats.yesPercent.toFixed(1)}% · No:{' '}
                    {col.yesNoStats.noPercent.toFixed(1)}%
                  </span>
                )}
              </div>
            ))}
            {keptColumns.length === 0 && (
              <p className="text-gray-500 italic">No columns are kept with the current thresholds.</p>
            )}
          </div>
        </div>
        <div>
          <h4 className="text-sm font-semibold text-neutral-darker mb-2 flex items-center">
            Dropped columns{' '}
            <span className="ml-2 text-xs text-gray-500">({droppedColumns.length})</span>
          </h4>
          <div className="space-y-1 max-h-40 overflow-y-auto border border-neutral-light rounded-md p-2 bg-white">
            {droppedColumns.map((col) => (
              <div
                key={col.column}
                className="flex justify-between items-center"
              >
                <span className="font-medium text-neutral-darker">{col.column}</span>
                <span className="text-gray-500">
                  {col.reason || 'Dropped based on thresholds'}
                </span>
              </div>
            ))}
            {droppedColumns.length === 0 && (
              <p className="text-gray-500 italic">
                No columns are dropped with the current thresholds.
              </p>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};

export default ColumnPlanSummary;

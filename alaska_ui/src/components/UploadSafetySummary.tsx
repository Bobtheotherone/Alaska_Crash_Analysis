import React from 'react';
import { ValidationResults } from './App';
import StatusPill from './ui/StatusPill';

interface UploadSafetySummaryProps {
  results: ValidationResults;
}

const CheckIcon: React.FC<{ className?: string }> = ({ className = 'h-5 w-5 text-green-500' }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    viewBox="0 0 20 20"
    fill="currentColor"
  >
    <path
      fillRule="evenodd"
      d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293A1 1 0 006.293 10.707l2 2a1 1 0 001.414 0l4-4z"
      clipRule="evenodd"
    />
  </svg>
);

const ErrorIcon: React.FC<{ className?: string }> = ({ className = 'h-5 w-5 text-red-500' }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    viewBox="0 0 20 20"
    fill="currentColor"
  >
    <path
      fillRule="evenodd"
      d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
      clipRule="evenodd"
    />
  </svg>
);

const UploadSafetySummary: React.FC<UploadSafetySummaryProps> = ({ results }) => {
  const ingestionStatus = results.ingestionOverallStatus;
  const ingestionMessage = results.ingestionMessage;
  const ingestionErrorCode = results.ingestionErrorCode;
  const ingestionRowChecks = results.ingestionRowChecks;

  const hasGenericError = !!results.error && !ingestionStatus;

  if (!ingestionStatus && !hasGenericError) {
    return null;
  }

  // Generic schema/validation error without an ingestion report
  if (!ingestionStatus && hasGenericError) {
    return (
      <section className="rounded-md bg-red-50 border border-red-200 text-red-800 px-4 py-3 text-sm">
        <div className="flex items-start gap-2">
          <ErrorIcon />
          <div>
            <p className="font-semibold">Validation error</p>
            <p className="mt-0.5">{results.error}</p>
          </div>
        </div>
      </section>
    );
  }

  const isAccepted = ingestionStatus === 'accepted';

  return (
    <section
      className={`rounded-md border px-4 py-3 text-sm ${
        isAccepted ? 'bg-green-50 border-green-200 text-green-900' : 'bg-red-50 border-red-200 text-red-900'
      }`}
    >
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div className="flex items-start gap-2">
          {isAccepted ? <CheckIcon /> : <ErrorIcon />}
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <p className="font-semibold">
                Ingestion gateway: {isAccepted ? 'accepted' : 'rejected'}
              </p>
              <StatusPill tone={isAccepted ? 'success' : 'error'}>
                {isAccepted ? 'Safe to proceed' : 'Needs attention'}
              </StatusPill>
            </div>
            <p className="mt-0.5">
              {ingestionMessage ||
                (isAccepted
                  ? 'The dataset passed the upload safety checks.'
                  : 'The dataset did not pass the upload safety checks.')}
            </p>
            {ingestionErrorCode && (
              <p className="mt-0.5 text-xs opacity-80">
                Error code:{' '}
                <code className="font-mono text-xs">{ingestionErrorCode}</code> – see the ingestion
                error-code reference for details.
              </p>
            )}
          </div>
        </div>
        {ingestionRowChecks && (
          <div className="text-xs sm:text-right space-y-0.5 opacity-80">
            <p>Rows seen: {ingestionRowChecks.total_rows.toLocaleString()}</p>
            <p>
              Invalid rows (schema/values):{' '}
              {ingestionRowChecks.invalid_row_count.toLocaleString()}
            </p>
            <p>
              Invalid rows (geometry):{' '}
              {ingestionRowChecks.invalid_geo_row_count.toLocaleString()}
            </p>
          </div>
        )}
      </div>
    </section>
  );
};

export default UploadSafetySummary;

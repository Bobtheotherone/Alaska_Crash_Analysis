import React, { useMemo, useState } from 'react';
import { ValidationResults } from './App';
import StatusPill from './ui/StatusPill';

interface IngestionLogAccordionProps {
  results: ValidationResults;
}

type IngestionStep = NonNullable<ValidationResults['ingestionSteps']>[number];

const ChevronIcon: React.FC<{ isOpen: boolean }> = ({ isOpen }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    className={`h-4 w-4 text-gray-500 transform transition-transform duration-200 ${
      isOpen ? '-rotate-180' : ''
    }`}
    fill="none"
    viewBox="0 0 24 24"
    stroke="currentColor"
  >
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
  </svg>
);

const IngestionLogAccordion: React.FC<IngestionLogAccordionProps> = ({ results }) => {
  const steps = results.ingestionSteps as IngestionStep[] | undefined;

  if (!steps || steps.length === 0) {
    return null;
  }

  const [isOpen, setIsOpen] = useState(
    steps.some((s) => s.status === 'failed' || s.severity === 'error')
  );

  const counts = useMemo(() => {
    let passed = 0;
    let failed = 0;
    let skipped = 0;
    steps.forEach((step) => {
      if (step.status === 'passed') passed += 1;
      else if (step.status === 'failed') failed += 1;
      else skipped += 1;
    });
    return { passed, failed, skipped };
  }, [steps]);

  const summaryTextParts: string[] = [];
  if (counts.passed) summaryTextParts.push(`${counts.passed} passed`);
  if (counts.failed) summaryTextParts.push(`${counts.failed} failed`);
  if (counts.skipped) summaryTextParts.push(`${counts.skipped} skipped`);
  const summaryText = summaryTextParts.join(' · ');

  return (
    <section className="border border-neutral-light rounded-md bg-white">
      <button
        type="button"
        className="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-gray-50"
        onClick={() => setIsOpen((open) => !open)}
      >
        <div className="flex flex-col sm:flex-row sm:items-baseline gap-1 sm:gap-2">
          <h3 className="text-sm font-semibold text-neutral-darker">
            Ingestion checks (contract &amp; safety)
          </h3>
          <p className="text-xs text-gray-500">{summaryText || 'No checks were run.'}</p>
        </div>
        <div className="flex items-center gap-2">
          {counts.failed > 0 ? (
            <StatusPill tone="error">{counts.failed} issue(s)</StatusPill>
          ) : (
            <StatusPill tone="success">All checks passed</StatusPill>
          )}
          <ChevronIcon isOpen={isOpen} />
        </div>
      </button>
      {isOpen && (
        <div className="border-t border-neutral-light overflow-x-auto">
          <table className="min-w-full text-xs">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-3 py-1.5 text-left font-semibold text-gray-700">Check</th>
                <th className="px-3 py-1.5 text-left font-semibold text-gray-700">Details</th>
                <th className="px-3 py-1.5 text-left font-semibold text-gray-700">Status</th>
                <th className="px-3 py-1.5 text-left font-semibold text-gray-700">Severity</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-light">
              {steps.map((step, idx) => (
                <tr key={`${step.step}-${idx}`} className="bg-white">
                  <td className="px-3 py-1.5 whitespace-nowrap align-top">
                    <div className="font-medium text-neutral-darker">{step.step}</div>
                    {step.code && (
                      <div className="font-mono text-[10px] uppercase text-gray-500 mt-0.5">
                        {step.code}
                      </div>
                    )}
                  </td>
                  <td className="px-3 py-1.5 text-gray-700 align-top">
                    {step.message}
                  </td>
                  <td className="px-3 py-1.5 align-top">
                    <StatusPill
                      tone={
                        step.status === 'passed'
                          ? 'success'
                          : step.status === 'failed'
                          ? 'error'
                          : 'info'
                      }
                    >
                      {step.status}
                    </StatusPill>
                  </td>
                  <td className="px-3 py-1.5 text-gray-700 align-top">
                    {step.severity === 'error'
                      ? 'Hard fail'
                      : step.severity === 'warning'
                      ? step.is_hard_fail
                        ? 'Warning (hard)'
                        : 'Warning'
                      : 'Info'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
};

export default IngestionLogAccordion;

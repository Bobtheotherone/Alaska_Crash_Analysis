import React, { ReactNode } from 'react';
import { ChevronDownIcon, QuestionCircleIcon } from '../constants';
import StatusIndicator from './StatusIndicator';

interface CollapsibleSectionProps {
  title: string;
  children: ReactNode;
  isOpen: boolean;
  onToggle: () => void;
  status?: { type: 'ready' | 'valid' | 'error'; text: string } | null;
}

const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({
  title,
  children,
  isOpen,
  onToggle,
  status,
}) => {
  return (
    <section className="border border-neutral-medium rounded-lg overflow-hidden bg-white">
      <h3 className="w-full">
        <button
          onClick={onToggle}
          className="w-full flex justify-between items-center px-4 py-3 text-left text-neutral-darker bg-neutral-light hover:bg-gray-200 transition-colors duration-200"
          aria-expanded={isOpen}
        >
          <div className="flex flex-col">
            <span className="text-sm font-semibold">{title}</span>
            {status && (
              <span className="mt-0.5 text-xs text-gray-500 truncate">
                {status.text}
              </span>
            )}
          </div>
          <div className="flex items-center space-x-2">
            {status && <StatusIndicator status={status} />}
            <QuestionCircleIcon />
            <ChevronDownIcon isOpen={isOpen} />
          </div>
        </button>
      </h3>
      {isOpen && (
        <div className="px-4 py-4 border-t border-neutral-medium bg-white">
          {children}
        </div>
      )}
    </section>
  );
};

export default CollapsibleSection;

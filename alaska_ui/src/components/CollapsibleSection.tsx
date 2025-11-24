
import React, { ReactNode } from 'react';
import { ChevronDownIcon, QuestionCircleIcon } from '../constants';
import StatusIndicator from './StatusIndicator';

interface CollapsibleSectionProps {
  title: string;
  children: ReactNode;
  isOpen: boolean;
  onToggle: () => void;
  status?: { type: 'ready' | 'valid' | 'error'; text: string; } | null;
}

const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({ title, children, isOpen, onToggle, status }) => {
  return (
    <div className="border border-neutral-medium rounded-lg">
      <h3 className="text-lg font-semibold w-full">
        <button
          onClick={onToggle}
          className="w-full flex justify-between items-center p-4 text-left text-neutral-darker bg-neutral-light hover:bg-gray-200 transition-colors duration-200 rounded-t-lg"
          aria-expanded={isOpen}
        >
          <span>{title}</span>
          <div className="flex items-center space-x-2">
            {status && <StatusIndicator status={status} />}
            <QuestionCircleIcon />
            <ChevronDownIcon isOpen={isOpen} />
          </div>
        </button>
      </h3>
      {isOpen && (
        <div className="p-4 border-t border-neutral-medium">
          {children}
        </div>
      )}
    </div>
  );
};

export default CollapsibleSection;

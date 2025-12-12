
import React from 'react';

interface StatusIndicatorProps {
  status: {
    type: 'ready' | 'valid' | 'error';
    text: string;
  };
}

const STATUS_CONFIG = {
  ready: { color: 'bg-yellow-400' },
  valid: { color: 'bg-green-500' },
  error: { color: 'bg-red-500' },
};

const StatusIndicator: React.FC<StatusIndicatorProps> = ({ status }) => {
  const config = STATUS_CONFIG[status.type];
  return (
    <div className="flex items-center space-x-2 mr-2">
      <span className={`h-3 w-3 rounded-full ${config.color} border border-gray-500/20`}></span>
      <span className="text-xs font-medium text-gray-600 hidden sm:inline">{status.text}</span>
    </div>
  );
};

export default StatusIndicator;

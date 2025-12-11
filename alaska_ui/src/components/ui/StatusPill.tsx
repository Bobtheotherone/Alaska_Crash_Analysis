import React from 'react';

export type StatusTone = 'neutral' | 'info' | 'success' | 'warning' | 'error';

interface StatusPillProps {
  tone?: StatusTone;
  children: React.ReactNode;
  className?: string;
}

const toneClasses: Record<StatusTone, string> = {
  neutral: 'bg-gray-100 text-gray-700 border-gray-200',
  info: 'bg-blue-50 text-blue-800 border-blue-200',
  success: 'bg-green-50 text-green-800 border-green-200',
  warning: 'bg-yellow-50 text-yellow-800 border-yellow-200',
  error: 'bg-red-50 text-red-800 border-red-200',
};

const StatusPill: React.FC<StatusPillProps> = ({ tone = 'neutral', children, className = '' }) => {
  const base =
    'inline-flex items-center rounded-full px-2.5 py-0.5 text-[11px] font-semibold border whitespace-nowrap';
  const classes = `${base} ${toneClasses[tone]} ${className}`;
  return <span className={classes}>{children}</span>;
};

export default StatusPill;

import React from 'react';

const styles = {
  organic: 'bg-emerald-100 text-emerald-800 ring-emerald-300',
  pesticide: 'bg-amber-100 text-amber-800 ring-amber-300',
  disease: 'bg-rose-100 text-rose-800 ring-rose-300',
  both: 'bg-violet-100 text-violet-800 ring-violet-300',
  unknown: 'bg-gray-100 text-gray-700 ring-gray-300',
};

const labels = {
  organic: 'Organic',
  pesticide: 'Pesticide',
  disease: 'Disease',
  both: 'Both',
};

export default function ClassBadge({ label, size = 'md' }) {
  const key = String(label || '').toLowerCase();
  const klass = styles[key] || styles.unknown;
  const text = labels[key] || (label ? String(label) : 'Unknown');
  const sizing = size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-3 py-1 text-sm';
  return (
    <span className={`inline-flex items-center gap-1 rounded-full font-medium ring-1 ${sizing} ${klass}`}>
      <span className="h-2 w-2 rounded-full bg-current opacity-70"></span>
      {text}
    </span>
  );
}

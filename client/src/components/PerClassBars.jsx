import React from 'react';

export default function PerClassBars({ probs }) {
  if (!probs || typeof probs !== 'object') return null;
  const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  const colors = ['bg-emerald-500','bg-sky-500','bg-mango-500','bg-violet-500','bg-rose-500','bg-amber-500'];
  return (
    <div className="mt-4 space-y-2">
      {entries.map(([label, pct], idx) => (
        <div key={label} className="flex items-center gap-3">
          <div className="w-32 text-xs text-gray-700 capitalize truncate">{label}</div>
          <div className="flex-1 h-2.5 rounded-full bg-gray-100 overflow-hidden">
            <div className={`h-full rounded-full ${colors[idx % colors.length]}`} style={{ width: `${Math.max(0, Math.min(100, pct))}%` }} />
          </div>
          <div className="w-14 text-right text-xs text-gray-800 font-medium">{pct.toFixed(1)}%</div>
        </div>
      ))}
    </div>
  );
}

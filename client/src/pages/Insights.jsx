import React, { useEffect, useMemo, useState } from 'react';
import ModelComparisonChart from '../components/ModelComparisonChart';

export default function Insights() {
  const [task, setTask] = useState('disease'); // 'disease' | 'pesticide'
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [modelKey, setModelKey] = useState('svm');
  const [metric, setMetric] = useState('f1-score');

  useEffect(() => {
    let mounted = true;
    setData(null);
    setError('');
    fetch(`/api/models/comparison?task=${task}`)
      .then(async (res) => {
        const ct = res.headers.get('content-type') || '';
        const payload = ct.includes('application/json') ? await res.json() : { error: await res.text() };
        if (!res.ok) throw new Error(payload?.error || 'Failed to load model comparison');
        if (mounted) setData(payload);
      })
      .catch((e) => mounted && setError(e.message));
    return () => { mounted = false; };
  }, [task]); // Re-fetch when task changes

  useEffect(() => {
    if (!data) return;
    const best = data?.best?.name;
    if (best) setModelKey(best);
  }, [data]);

  const classNames = data?.models?.class_names || [];
  const report = data?.models?.[modelKey]?.report || null;
  const confusion = data?.models?.[modelKey]?.confusion_matrix || null;

  const perClass = useMemo(() => {
    if (!report) return [];
    // Report keys are class indices as strings; map to class names
    return classNames.map((name, idx) => {
      const r = report[String(idx)] || {};
      return {
        name,
        precision: Number(r['precision'] || 0) * 100,
        recall: Number(r['recall'] || 0) * 100,
        f1: Number(r['f1-score'] || 0) * 100,
        support: Number(r['support'] || 0),
      };
    });
  }, [report, classNames]);

  const metricKey = metric === 'precision' ? 'precision' : metric === 'recall' ? 'recall' : 'f1';

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h1 className="text-2xl md:text-3xl font-semibold">Model insights</h1>
        <p className="text-gray-600">Compare validation metrics and inspect confusion matrices for each model.</p>
      </div>

      {/* Task Selector */}
      <div className="card p-4 bg-white/70 ring-1 ring-amber-100/60">
        <label className="block text-sm font-medium text-gray-800 mb-3">Analysis Type</label>
        <div className="flex gap-4">
          <button
            type="button"
            onClick={() => setTask('disease')}
            className={`flex-1 px-4 py-2 rounded-xl font-medium transition-all ${
              task === 'disease'
                ? 'bg-gradient-to-r from-mango-500 to-amber-500 text-white shadow-lg'
                : 'bg-white/60 backdrop-blur text-gray-700 hover:bg-white/80'
            }`}
          >
            🦠 Disease Detection
          </button>
          <button
            type="button"
            onClick={() => setTask('pesticide')}
            className={`flex-1 px-4 py-2 rounded-xl font-medium transition-all ${
              task === 'pesticide'
                ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white shadow-lg'
                : 'bg-white/60 backdrop-blur text-gray-700 hover:bg-white/80'
            }`}
          >
            🧪 Pesticide Detection
          </button>
        </div>
      </div>

      <ModelComparisonChart task={task} />

      {error && (
        <div className="card border-red-100 bg-red-50 p-4 text-red-700">{error}</div>
      )}

      {data && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="card p-6 bg-white/70 ring-1 ring-amber-100/60">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-lg font-semibold">Per-class metrics</h3>
              <div className="flex items-center gap-2">
                <select value={modelKey} onChange={(e) => setModelKey(e.target.value)} className="rounded-md border-gray-200 text-sm">
                  {Object.keys(data.models).filter(k => k !== 'class_names').map(k => (
                    <option key={k} value={k}>{k.toUpperCase()}</option>
                  ))}
                </select>
                <select value={metric} onChange={(e) => setMetric(e.target.value)} className="rounded-md border-gray-200 text-sm">
                  <option value="f1-score">F1-score</option>
                  <option value="precision">Precision</option>
                  <option value="recall">Recall</option>
                </select>
              </div>
            </div>
            <div className="space-y-2">
              {perClass.map((c) => (
                <div key={c.name} className="flex items-center gap-3">
                  <div className="w-32 text-xs text-gray-600 truncate">{c.name}</div>
                  <div className="flex-1 h-2 rounded-full bg-gray-100 overflow-hidden">
                    <div className="h-full rounded-full bg-sky-400" style={{ width: `${Math.max(0, Math.min(100, c[metricKey]))}%` }} />
                  </div>
                  <div className="w-14 text-right text-xs text-gray-700">{c[metricKey].toFixed(1)}%</div>
                </div>
              ))}
            </div>
          </div>

          <div className="card p-6 bg-white/70 ring-1 ring-amber-100/60 overflow-x-auto">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-lg font-semibold">Confusion matrix</h3>
              <span className="inline-flex items-center gap-2 rounded-full bg-amber-100 px-3 py-1 text-xs font-medium text-amber-800">{modelKey.toUpperCase()}</span>
            </div>
            {confusion ? (
              <table className="min-w-full text-sm">
                <thead>
                  <tr>
                    <th className="p-2"></th>
                    {classNames.map((c) => (
                      <th key={c} className="p-2 text-left font-medium text-gray-600">Pred: {c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {confusion.map((row, i) => (
                    <tr key={i} className="odd:bg-white/60 even:bg-white/30">
                      <td className="p-2 font-medium text-gray-700">True: {classNames[i]}</td>
                      {row.map((v, j) => (
                        <td key={j} className="p-2">
                          <div className="w-16 text-center rounded bg-gray-100 text-gray-700">{v}</div>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="text-sm text-gray-500">No confusion matrix available for this model.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

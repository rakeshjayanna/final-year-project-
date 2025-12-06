import React, { useEffect, useRef, useState } from 'react';
import ClassBadge from '../components/ClassBadge';
import { useLocation } from 'react-router-dom';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

export default function Compare() {
  const location = useLocation();
  const inputRef = useRef(null);
  const [task, setTask] = useState('disease'); // 'disease' | 'pesticide'
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [tab, setTab] = useState('cnn');

  // Load result passed from detect or from sessionStorage if present
  useEffect(() => {
    const passed = location.state && location.state.compareResult ? location.state.compareResult : null;
    if (passed) {
      setResult(passed);
      try { sessionStorage.setItem('lastCompare', JSON.stringify(passed)); } catch {}
    } else {
      try {
        const saved = sessionStorage.getItem('lastCompare');
        if (saved) setResult(JSON.parse(saved));
      } catch {}
    }
  }, [location.state]);

  const MAX_BYTES = 5 * 1024 * 1024;
  const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

  const onFile = (f) => {
    setError('');
    setResult(null);
    if (!f) { setFile(null); setPreview(''); return; }
    if (!ACCEPTED_TYPES.includes(f.type)) {
      setError('Only JPG, PNG, or WEBP images are allowed.');
      return;
    }
    if (f.size > MAX_BYTES) {
      setError('File is too large (max 5MB).');
      return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    if (!file) { setError('Please choose an image.'); return; }
    setLoading(true); setError(''); setResult(null);
    const form = new FormData();
    form.append('image', file);
    form.append('task', task); // Add task to form data
    try {
      const res = await fetch('/api/compare-image', { method: 'POST', body: form, headers: { 'Accept': 'application/json' } });
      const ct = res.headers.get('content-type') || '';
      const data = ct.includes('application/json') ? await res.json() : { error: await res.text() };
      if (!res.ok) throw new Error(data?.error || 'Request failed');
      setResult(data);
    } catch (err) {
      setError(err?.message || 'Request failed');
    } finally {
      setLoading(false);
    }
  };

  const chart = result ? (() => {
    const entries = Object.entries(result.models || {}).map(([k, v]) => ({
      key: k,
      label: k.replace('_', ' ').toUpperCase(),
      conf: v?.confidence || 0,
    }));
    return {
      labels: entries.map((e) => e.label),
      datasets: [{
        label: 'Confidence (%)',
        data: entries.map((e) => e.conf),
        backgroundColor: ['#f59e0b', '#60a5fa', '#34d399', '#a78bfa'].slice(0, entries.length),
        borderRadius: 6,
      }],
    };
  })() : null;

  const options = {
    responsive: true,
    plugins: {
      legend: { position: 'top' },
      title: { display: true, text: 'Per-image confidence by model' },
      tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toFixed(2)}%` } },
    },
    scales: { y: { beginAtZero: true, max: 100, ticks: { callback: (v) => `${v}%` } } },
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h1 className="text-2xl md:text-3xl font-semibold">Compare models on your image</h1>
  <p className="text-gray-600">We run CNN and SVM and pick the one with best validation accuracy. If tied, highest confidence on your image wins.</p>
      </div>

      {!result && (
      <form onSubmit={onSubmit} className="card p-6 bg-gradient-to-br from-white to-amber-50 shadow-xl ring-1 ring-amber-100/60">
        {/* Task Selector */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-800 mb-3">Analysis Type</label>
          <div className="flex gap-4">
            <button
              type="button"
              onClick={() => setTask('disease')}
              className={`flex-1 px-4 py-2 rounded-xl font-medium transition-all text-sm ${
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
              className={`flex-1 px-4 py-2 rounded-xl font-medium transition-all text-sm ${
                task === 'pesticide'
                  ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white shadow-lg'
                  : 'bg-white/60 backdrop-blur text-gray-700 hover:bg-white/80'
              }`}
            >
              🧪 Pesticide Detection
            </button>
          </div>
        </div>
        
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1">
            <label className="block text-sm font-medium text-gray-700 mb-2">Upload image</label>
            <input ref={inputRef} type="file" accept="image/*" onChange={(e) => onFile(e.target.files?.[0] || null)}
              className="block w-full text-sm text-gray-700 file:mr-4 file:rounded-md file:border-0 file:bg-mango-100 file:px-4 file:py-2 file:text-mango-700 hover:file:bg-mango-200" />
          </div>
          <div className="w-full md:w-64">
            <div className="aspect-square overflow-hidden rounded-xl border border-amber-100 bg-white shadow-inner flex items-center justify-center">
              {preview ? (
                <img src={preview} alt="preview" className="h-full w-full object-cover" />
              ) : (
                <div className="text-gray-400">No image</div>
              )}
            </div>
          </div>
        </div>
        <div className="mt-4">
          <button type="submit" className="btn-primary" disabled={loading}>{loading ? 'Comparing…' : 'Compare models'}</button>
        </div>
      </form>
      )}

      {error && (
        <div className="card bg-red-50 border-red-100 text-red-700 p-4">{error}</div>
      )}

      {result && (
        <div className="space-y-4">
          <div className="card p-4 bg-white/70 ring-1 ring-amber-100/60">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">Chosen model</p>
                <p className="text-xl font-semibold capitalize">{result.selection?.model?.replace('_',' ') || '—'}</p>
              </div>
              <div className="text-right">
                <p className="text-sm text-gray-500">Reason</p>
                <p className="text-sm">{result.selection?.reason || '—'}</p>
              </div>
            </div>
            {result.selection?.detail && (
              <p className="mt-2 text-xs text-gray-500">
                Validation accuracy — CNN: {(result.selection.detail.cnn_acc*100).toFixed(1)}% · SVM: {(result.selection.detail.svm_acc*100).toFixed(1)}%
              </p>
            )}
          </div>

          <div className="card p-6 bg-white/70 shadow-lg ring-1 ring-amber-100/60">
            <Bar data={chart} options={options} height={120} />
          </div>

          <div className="card p-4 bg-white/70 ring-1 ring-amber-100/60">
            <p className="text-sm text-gray-500">Final prediction</p>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <ClassBadge label={result.final?.label} />
              </div>
              <p className="text-xl font-semibold text-mango-700">{typeof result.final?.confidence === 'number' ? `${result.final.confidence}%` : '—'}</p>
            </div>
          </div>

          {/* Per-class probabilities */}
          {result.selection?.model && result.models?.[result.selection.model]?.probs && (
            <div className="card p-6 bg-white/70 ring-1 ring-amber-100/60">
              <h3 className="text-lg font-semibold mb-3">Per-class confidence</h3>
              {(() => {
                const PerClassBars = require('../components/PerClassBars').default;
                const dist = result.models[result.selection.model]?.probs;
                if (!dist) return null;
                return <PerClassBars probs={dist} />;
              })()}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

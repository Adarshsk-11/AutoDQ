import { useState } from 'react'
import { cleanDataset, downloadUrl, ApiError } from '../services/api.js'
import ErrorBanner from '../components/ErrorBanner.jsx'
import Spinner from '../components/Spinner.jsx'

const SEVERITY_ORDER = ['critical', 'high', 'medium', 'low']
const SEVERITY_BADGE = {
  critical: 'bg-red-600',
  high: 'bg-orange-500',
  medium: 'bg-amber-500',
  low: 'bg-slate-400',
}

export default function RecommendationsPage({ dataset, analysis }) {
  const { recommendations } = analysis

  const sorted = [...recommendations].sort(
    (a, b) => SEVERITY_ORDER.indexOf(a.severity) - SEVERITY_ORDER.indexOf(b.severity)
  )

  return (
    <div className="space-y-3">
      {recommendations.length === 0 ? (
        <div className="bg-white border border-slate-200 rounded-2xl p-10 text-center text-slate-500">
          No recommendations — no data quality issues were detected to act on.
        </div>
      ) : (
        <>
          <p className="text-sm text-slate-500 mb-4">
            Prioritized by severity. Each recommendation is generated from an issue actually detected in this dataset.
          </p>
          {sorted.map((rec, i) => (
            <div key={i} className="bg-white border border-slate-200 rounded-xl p-4 flex gap-4 items-start">
              <span className={`mt-1 shrink-0 w-2.5 h-2.5 rounded-full ${SEVERITY_BADGE[rec.severity]}`} />
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs uppercase tracking-wide text-slate-400 font-semibold">{rec.severity}</span>
                  {rec.columns?.length > 0 && (
                    <span className="text-xs text-slate-400 font-mono">{rec.columns.join(', ')}</span>
                  )}
                </div>
                <p className="text-sm text-slate-800">{rec.message}</p>
              </div>
            </div>
          ))}
        </>
      )}

      <AutoFixPanel datasetId={dataset?.dataset_id} />
    </div>
  )
}

function AutoFixPanel({ datasetId }) {
  const [status, setStatus] = useState('idle') // idle | cleaning | done | error
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  const handleClean = async () => {
    setStatus('cleaning')
    setError('')
    try {
      const res = await cleanDataset(datasetId)
      setResult(res)
      setStatus('done')
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not generate the corrected dataset.')
      setStatus('error')
    }
  }

  return (
    <div className="mt-8 bg-emerald-50 border border-emerald-200 rounded-2xl p-6">
      <h3 className="text-sm font-semibold text-emerald-800 uppercase tracking-wide mb-1">
        Auto-Fix: Generate a Corrected Dataset
      </h3>
      <p className="text-xs text-emerald-700 mb-4">
        Algorithmically fixes Completeness, Consistency, Validity, Uniqueness, and Balance —
        median/mode imputation, casing/whitespace normalization, duplicate removal, and
        class-conditional resampling for imbalance. Never touches columns flagged as redundant
        or leaked — those remain visible in Issues for you to review, not silently deleted.
      </p>

      {status !== 'done' && (
        <button
          onClick={handleClean}
          disabled={status === 'cleaning' || !datasetId}
          className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-300 text-white text-sm font-medium flex items-center gap-2"
        >
          {status === 'cleaning' ? <Spinner label="Generating corrected dataset…" /> : 'Download Corrected Dataset'}
        </button>
      )}

      <div className="mt-3">
        <ErrorBanner message={error} onRetry={status === 'error' ? handleClean : undefined} />
      </div>

      {result && (
        <div className="mt-4 space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-white rounded-xl p-4 text-center border border-slate-200">
              <div className="text-xs text-slate-400 uppercase tracking-wide mb-1">Before</div>
              <div className="text-2xl font-bold text-slate-700">{result.before.dqi.dqi_0_100}</div>
              <div className="text-xs text-slate-400">{result.before.n_rows.toLocaleString()} rows</div>
            </div>
            <div className="bg-white rounded-xl p-4 text-center border border-emerald-300">
              <div className="text-xs text-emerald-600 uppercase tracking-wide mb-1">After</div>
              <div className="text-2xl font-bold text-emerald-700">{result.after.dqi.dqi_0_100}</div>
              <div className="text-xs text-slate-400">{result.after.n_rows.toLocaleString()} rows</div>
            </div>
          </div>

          <div>
            <h4 className="text-xs font-semibold text-emerald-800 uppercase tracking-wide mb-2">What was fixed</h4>
            <ul className="text-xs text-slate-600 space-y-1 bg-white rounded-xl border border-slate-200 p-3 max-h-48 overflow-y-auto">
              {result.fix_log.map((line, i) => (
                <li key={i}>• {line}</li>
              ))}
            </ul>
          </div>

          <a
            href={downloadUrl(result.cleaned_dataset_id)}
            className="inline-block px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-medium"
          >
            Download {result.cleaned_filename}
          </a>
        </div>
      )}
    </div>
  )
}

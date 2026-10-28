import { useEffect, useState } from 'react'
import { uploadDataset, analyzeDataset, listDatasets, getDatasetInfo, ApiError } from '../services/api.js'
import ErrorBanner from '../components/ErrorBanner.jsx'
import Spinner from '../components/Spinner.jsx'

const DOMAINS = ['general', 'healthcare', 'finance', 'ecommerce']

export default function UploadPage({ onAnalyzed }) {
  const [file, setFile] = useState(null)
  const [uploadInfo, setUploadInfo] = useState(null)
  const [targetColumn, setTargetColumn] = useState('')
  const [domain, setDomain] = useState('general')
  const [status, setStatus] = useState('idle') // idle | uploading | uploaded | analyzing | error
  const [error, setError] = useState('')
  const [recent, setRecent] = useState([])
  const [recentError, setRecentError] = useState('')
  const [recentLoading, setRecentLoading] = useState(true)

  useEffect(() => {
    loadRecent()
  }, [])

  async function loadRecent() {
    setRecentLoading(true)
    setRecentError('')
    try {
      const list = await listDatasets()
      setRecent(list)
    } catch (err) {
      // Non-fatal: the recent-datasets panel is a convenience, not a
      // blocker, so we show a small inline note rather than a full-page error.
      setRecentError(err instanceof ApiError ? err.message : 'Could not load recent datasets.')
    } finally {
      setRecentLoading(false)
    }
  }

  const handleFileChange = async (e) => {
    const f = e.target.files?.[0]
    if (!f) return
    setFile(f)
    setError('')
    setStatus('uploading')
    try {
      const info = await uploadDataset(f)
      setUploadInfo(info)
      setStatus('uploaded')
      loadRecent()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Upload failed unexpectedly.')
      setStatus('error')
    }
  }

  const handleAnalyze = async () => {
    if (!uploadInfo) return
    setStatus('analyzing')
    setError('')
    try {
      const result = await analyzeDataset(uploadInfo.dataset_id, { targetColumn, domain })
      onAnalyzed(uploadInfo, result)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Analysis failed unexpectedly.')
      setStatus('uploaded')
    }
  }

  const handleReopenRecent = async (datasetId) => {
    setError('')
    setStatus('uploading')
    try {
      const info = await getDatasetInfo(datasetId)
      setUploadInfo(info)
      setTargetColumn(info.last_target_column || '')
      setFile({ name: info.filename })
      setStatus('uploaded')
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not reopen that dataset.')
      setStatus('error')
    }
  }

  return (
    <main className="flex-1 flex flex-col items-center px-6 py-16 gap-8">
      <div className="max-w-xl w-full bg-white border border-slate-200 rounded-2xl shadow-sm p-8">
        <h2 className="text-2xl font-bold mb-1">Upload a dataset</h2>
        <p className="text-slate-500 text-sm mb-6">Supported: CSV · XLSX · JSON</p>

        <label className="block border-2 border-dashed border-slate-300 rounded-xl p-8 text-center cursor-pointer hover:border-brand-500 transition mb-6">
          <input type="file" accept=".csv,.xlsx,.xls,.json" className="hidden" onChange={handleFileChange} />
          {!file && <span className="text-slate-500">Click to choose a file, or drag one here</span>}
          {file && <span className="text-slate-800 font-medium">{file.name}</span>}
        </label>

        {status === 'uploading' && <div className="mb-4"><Spinner label="Uploading and validating…" /></div>}

        <div className="mb-4">
          <ErrorBanner
            message={error}
            onRetry={file ? () => handleFileChange({ target: { files: [file] } }) : undefined}
          />
        </div>

        {uploadInfo && (
          <div className="space-y-4 mb-6">
            <div className="text-sm text-slate-600 bg-slate-50 rounded-lg px-4 py-3">
              {uploadInfo.n_rows.toLocaleString()} rows × {uploadInfo.n_columns} columns
            </div>

            {uploadInfo.warnings?.length > 0 && (
              <div className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-4 py-3 space-y-1">
                {uploadInfo.warnings.map((w, i) => (
                  <div key={i}>⚠ {w}</div>
                ))}
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Target column <span className="text-slate-400 font-normal">(optional — auto-detected if left blank)</span>
              </label>
              <select
                value={targetColumn}
                onChange={(e) => setTargetColumn(e.target.value)}
                className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm"
              >
                <option value="">Auto-detect</option>
                {uploadInfo.columns.map((c) => (
                  <option key={c.name} value={c.name}>
                    {c.name} ({c.dtype})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Domain preset <span className="text-slate-400 font-normal">(affects DQI weights — configurable heuristics, not universal standards)</span>
              </label>
              <select
                value={domain}
                onChange={(e) => setDomain(e.target.value)}
                className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm"
              >
                {DOMAINS.map((d) => (
                  <option key={d} value={d}>
                    {d[0].toUpperCase() + d.slice(1)}
                  </option>
                ))}
              </select>
            </div>
          </div>
        )}

        <button
          disabled={!uploadInfo || status === 'analyzing'}
          onClick={handleAnalyze}
          className="w-full bg-brand-600 hover:bg-brand-700 disabled:bg-slate-300 text-white font-medium rounded-lg py-2.5 transition flex items-center justify-center gap-2"
        >
          {status === 'analyzing' ? <Spinner label="Analyzing…" /> : 'Analyze Dataset'}
        </button>
      </div>

      <div className="max-w-xl w-full">
        <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-2">
          Recent datasets this session
        </h3>
        {recentLoading && <Spinner label="Loading…" />}
        {!recentLoading && recentError && <p className="text-xs text-slate-400">{recentError}</p>}
        {!recentLoading && !recentError && recent.length === 0 && (
          <p className="text-xs text-slate-400">Nothing uploaded yet in this session.</p>
        )}
        {!recentLoading && recent.length > 0 && (
          <div className="bg-white border border-slate-200 rounded-xl divide-y divide-slate-100">
            {recent.map((r) => (
              <button
                key={r.dataset_id}
                onClick={() => handleReopenRecent(r.dataset_id)}
                className="w-full text-left px-4 py-3 flex items-center justify-between hover:bg-slate-50 transition"
              >
                <div>
                  <div className="text-sm font-medium text-slate-800">{r.filename}</div>
                  <div className="text-xs text-slate-400">
                    {r.n_rows.toLocaleString()} rows × {r.n_columns} cols
                    {r.analyzed && r.dqi_0_100 !== null && ` · DQI ${r.dqi_0_100}`}
                  </div>
                </div>
                {r.analyzed && (
                  <span className="text-xs px-2 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                    Analyzed
                  </span>
                )}
              </button>
            ))}
          </div>
        )}
      </div>
    </main>
  )
}

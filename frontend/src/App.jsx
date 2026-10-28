import { useState } from 'react'
import UploadPage from './pages/UploadPage.jsx'
import OverviewPage from './pages/OverviewPage.jsx'
import IssuesPage from './pages/IssuesPage.jsx'
import ProfilePage from './pages/ProfilePage.jsx'
import ModelPage from './pages/ModelPage.jsx'
import RecommendationsPage from './pages/RecommendationsPage.jsx'
import Nav from './components/Nav.jsx'
import ErrorBoundary from './components/ErrorBoundary.jsx'
import { reportUrl } from './services/api.js'

const PAGES = ['Overview', 'Issues', 'Dataset Profile', 'Model Readiness', 'Recommendations']

export default function App() {
  const [dataset, setDataset] = useState(null) // { dataset_id, filename, n_rows, n_columns, columns, warnings }
  const [analysis, setAnalysis] = useState(null)
  const [activePage, setActivePage] = useState('Overview')

  const handleAnalyzed = (uploadInfo, analysisResult) => {
    setDataset(uploadInfo)
    setAnalysis(analysisResult)
    setActivePage('Overview')
  }

  const reset = () => {
    setDataset(null)
    setAnalysis(null)
  }

  if (!analysis) {
    return (
      <div className="min-h-screen flex flex-col">
        <Header onReset={reset} showReset={false} />
        <ErrorBoundary onReset={reset}>
          <UploadPage onAnalyzed={handleAnalyzed} />
        </ErrorBoundary>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header onReset={reset} showReset={true} filename={dataset?.filename} datasetId={dataset?.dataset_id} />
      <Nav pages={PAGES} active={activePage} onChange={setActivePage} />
      <main className="flex-1 max-w-6xl mx-auto w-full px-6 py-8">
        <ErrorBoundary onReset={reset}>
          {activePage === 'Overview' && <OverviewPage dataset={dataset} analysis={analysis} />}
          {activePage === 'Issues' && <IssuesPage analysis={analysis} />}
          {activePage === 'Dataset Profile' && <ProfilePage analysis={analysis} />}
          {activePage === 'Model Readiness' && <ModelPage analysis={analysis} />}
          {activePage === 'Recommendations' && <RecommendationsPage dataset={dataset} analysis={analysis} />}
        </ErrorBoundary>
      </main>
      <footer className="text-center text-xs text-slate-400 py-6">
        AutoDQ MVP — analysis is generated live from the uploaded dataset, not hardcoded.
      </footer>
    </div>
  )
}

function Header({ onReset, showReset, filename, datasetId }) {
  return (
    <header className="bg-white border-b border-slate-200">
      <div className="max-w-6xl mx-auto w-full px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-brand-700">AutoDQ</h1>
          <p className="text-xs text-slate-500">Dataset Quality & Reliability Analyzer</p>
        </div>
        {showReset && (
          <div className="flex items-center gap-3">
            {filename && <span className="text-sm text-slate-500 hidden sm:inline">{filename}</span>}
            <a
              href={reportUrl(datasetId, 'html')}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm px-3 py-1.5 rounded-md border border-slate-300 hover:bg-slate-100 transition"
              title="Opens a printable report — use your browser's Print > Save as PDF to export a PDF"
            >
              Export Report
            </a>
            <a
              href={reportUrl(datasetId, 'json')}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm px-3 py-1.5 rounded-md border border-slate-300 hover:bg-slate-100 transition"
            >
              Export JSON
            </a>
            <button
              onClick={onReset}
              className="text-sm px-3 py-1.5 rounded-md border border-slate-300 hover:bg-slate-100 transition"
            >
              Analyze a new dataset
            </button>
          </div>
        )}
      </div>
    </header>
  )
}

import MissingValuesChart from '../charts/MissingValuesChart.jsx'
import HistogramChart from '../charts/HistogramChart.jsx'
import CategoryDistributionChart from '../charts/CategoryDistributionChart.jsx'

export default function ProfilePage({ analysis }) {
  const { profile, completeness, uniqueness } = analysis
  const redundancyIssues = analysis.issues.filter((i) => i.issue_type === 'redundant_features')

  return (
    <div className="space-y-8">
      <section className="grid sm:grid-cols-4 gap-4">
        <Stat label="Rows" value={profile.n_rows.toLocaleString()} />
        <Stat label="Columns" value={profile.n_columns} />
        <Stat label="Memory" value={formatBytes(profile.memory_bytes)} />
        <Stat label="Duplicate rows" value={`${uniqueness.duplicate_rows.toLocaleString()} (${uniqueness.duplicate_pct}%)`} />
      </section>

      <section>
        <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-3">Column types</h3>
        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-slate-500 text-xs uppercase">
              <tr>
                <th className="text-left px-4 py-2">Column</th>
                <th className="text-left px-4 py-2">Detected Type</th>
                <th className="text-left px-4 py-2">Missing %</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(profile.column_types).map(([col, type]) => (
                <tr key={col} className="border-t border-slate-100">
                  <td className="px-4 py-2 font-mono text-xs">{col}</td>
                  <td className="px-4 py-2 capitalize">{type}</td>
                  <td className="px-4 py-2">{completeness.per_column_missing_pct[col] ?? 0}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-3">Missing values by column</h3>
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <MissingValuesChart perColumnMissingPct={completeness.per_column_missing_pct} />
        </div>
      </section>

      {profile.numeric_profiles?.length > 0 && (
        <section>
          <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-3">Numeric distributions</h3>
          <div className="grid md:grid-cols-2 gap-4">
            {profile.numeric_profiles.map((p) => (
              <div key={p.column} className="bg-white border border-slate-200 rounded-xl p-4">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-sm">{p.column}</h4>
                  <span className="text-xs text-slate-400">skew: {p.skewness}</span>
                </div>
                <HistogramChart histogram={p.histogram} />
                <div className="grid grid-cols-4 gap-2 text-xs text-slate-500 mt-2">
                  <span>mean: {p.mean}</span>
                  <span>std: {p.std}</span>
                  <span>min: {p.min}</span>
                  <span>max: {p.max}</span>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {profile.categorical_profiles?.length > 0 && (
        <section>
          <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-3">Categorical distributions</h3>
          <div className="grid md:grid-cols-2 gap-4">
            {profile.categorical_profiles.map((p) => (
              <div key={p.column} className="bg-white border border-slate-200 rounded-xl p-4">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-sm">{p.column}</h4>
                  <span className="text-xs text-slate-400">{p.n_unique} unique</span>
                </div>
                <CategoryDistributionChart topValues={p.top_values} />
              </div>
            ))}
          </div>
        </section>
      )}

      <section>
        <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-3">Correlated / redundant features</h3>
        {redundancyIssues.length === 0 ? (
          <p className="text-sm text-slate-500">No highly correlated numeric feature pairs (threshold 0.90) detected.</p>
        ) : (
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 text-slate-500 text-xs uppercase">
                <tr>
                  <th className="text-left px-4 py-2">Feature A</th>
                  <th className="text-left px-4 py-2">Feature B</th>
                  <th className="text-left px-4 py-2">Correlation</th>
                  <th className="text-left px-4 py-2">Severity</th>
                </tr>
              </thead>
              <tbody>
                {redundancyIssues.map((issue, i) => (
                  <tr key={i} className="border-t border-slate-100">
                    <td className="px-4 py-2 font-mono text-xs">{issue.columns[0]}</td>
                    <td className="px-4 py-2 font-mono text-xs">{issue.columns[1]}</td>
                    <td className="px-4 py-2">{issue.measured_value}</td>
                    <td className="px-4 py-2 capitalize">{issue.severity}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}

function Stat({ label, value }) {
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 text-center">
      <div className="text-xl font-bold text-slate-800">{value}</div>
      <div className="text-xs text-slate-500 mt-1">{label}</div>
    </div>
  )
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

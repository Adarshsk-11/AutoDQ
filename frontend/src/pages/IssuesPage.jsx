const SEVERITY_ORDER = ['critical', 'high', 'medium', 'low']
const SEVERITY_STYLE = {
  critical: { label: 'Critical', bg: 'bg-red-50', border: 'border-red-200', badge: 'bg-red-600' },
  high: { label: 'High', bg: 'bg-orange-50', border: 'border-orange-200', badge: 'bg-orange-500' },
  medium: { label: 'Medium', bg: 'bg-amber-50', border: 'border-amber-200', badge: 'bg-amber-500' },
  low: { label: 'Low', bg: 'bg-slate-50', border: 'border-slate-200', badge: 'bg-slate-400' },
}

const TYPE_LABELS = {
  missing_values: 'Missing Values',
  duplicate_rows: 'Duplicate Rows',
  class_imbalance: 'Class Imbalance',
  outliers: 'Outliers',
  redundant_features: 'Redundant / Correlated Features',
  potential_leakage: 'Potential Data Leakage',
  potential_multivariate_leakage: 'Potential Multivariate Leakage',
  distributional_imbalance: 'Distributional Imbalance',
  constant_column: 'Constant Column (Zero Information)',
}

export default function IssuesPage({ analysis }) {
  const { issues } = analysis

  if (issues.length === 0) {
    return (
      <div className="bg-white border border-slate-200 rounded-2xl p-10 text-center text-slate-500">
        No issues detected. This dataset passed all AutoDQ checks.
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {SEVERITY_ORDER.map((sev) => {
        const group = issues.filter((i) => i.severity === sev)
        if (group.length === 0) return null
        const style = SEVERITY_STYLE[sev]
        return (
          <section key={sev}>
            <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-3 flex items-center gap-2">
              <span className={`inline-block w-2.5 h-2.5 rounded-full ${style.badge}`} />
              {style.label} ({group.length})
            </h3>
            <div className="space-y-3">
              {group.map((issue, idx) => (
                <div key={idx} className={`border rounded-xl p-4 ${style.bg} ${style.border}`}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-semibold text-sm text-slate-800">{TYPE_LABELS[issue.issue_type] || issue.issue_type}</span>
                    {issue.columns?.length > 0 && (
                      <span className="text-xs text-slate-500 font-mono">{issue.columns.join(', ')}</span>
                    )}
                  </div>
                  <p className="text-sm text-slate-700">{issue.explanation}</p>
                  {issue.measured_value !== null && issue.measured_value !== undefined && (
                    <p className="text-xs text-slate-500 mt-1">Measured value: {issue.measured_value}</p>
                  )}
                </div>
              ))}
            </div>
          </section>
        )
      })}
    </div>
  )
}

const CLASSIFICATION_METRIC_LABELS = { accuracy: 'Accuracy', precision: 'Precision', recall: 'Recall', f1: 'F1 Score' }
const REGRESSION_METRIC_LABELS = { mae: 'MAE', rmse: 'RMSE', r2: 'R²' }

const READINESS_STYLE = {
  High: { bg: 'bg-emerald-50', border: 'border-emerald-200', text: 'text-emerald-700' },
  Moderate: { bg: 'bg-amber-50', border: 'border-amber-200', text: 'text-amber-700' },
  Low: { bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-700' },
}

export default function ModelPage({ analysis }) {
  const { baseline } = analysis

  if (!baseline.available) {
    return (
      <div className="bg-white border border-slate-200 rounded-2xl p-10 text-center">
        <p className="text-slate-600 font-medium mb-2">Baseline model performance unavailable</p>
        <p className="text-sm text-slate-500">{baseline.reason_unavailable}</p>
      </div>
    )
  }

  const metricLabels = baseline.task_type === 'classification' ? CLASSIFICATION_METRIC_LABELS : REGRESSION_METRIC_LABELS
  const style = READINESS_STYLE[baseline.readiness] || READINESS_STYLE.Moderate
  const hasComparison = !!baseline.cleaned_features

  return (
    <div className="space-y-6">
      <section className={`rounded-2xl border p-6 ${style.bg} ${style.border}`}>
        <p className="text-xs uppercase tracking-wide text-slate-500 mb-1">Estimated Dataset Learnability</p>
        <div className="flex items-center gap-3">
          <span className={`text-3xl font-bold ${style.text}`}>{baseline.readiness}</span>
          <span className="text-sm text-slate-500">
            {baseline.task_type === 'classification' ? 'Classification' : 'Regression'} task, target: <span className="font-mono">{baseline.target_column}</span>
          </span>
        </div>
        <p className="text-xs text-slate-500 mt-2">
          {baseline.n_rows_used.toLocaleString()} rows used · {baseline.cv_folds}-fold cross-validation
          {hasComparison && ' · readiness based on the leakage/redundancy-excluded run below'}
        </p>
      </section>

      {hasComparison && (
        <section className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-sm text-amber-800">
          <strong>{baseline.excluded_columns.length} column(s)</strong> flagged as potential leakage or redundant
          features (<span className="font-mono text-xs">{baseline.excluded_columns.join(', ')}</span>) were excluded
          from the second run below, so you can see how much they were inflating performance.
        </section>
      )}

      <FeatureSetSection
        title="All features"
        subtitle={hasComparison ? 'Includes columns flagged as potential leakage or redundant' : undefined}
        featureSet={baseline.full_features}
        metricLabels={metricLabels}
        highlight={!hasComparison}
      />

      {hasComparison && (
        <FeatureSetSection
          title="Excluding flagged leakage / redundant features"
          subtitle="The more trustworthy estimate of real-world performance"
          featureSet={baseline.cleaned_features}
          metricLabels={metricLabels}
          highlight
        />
      )}

      <p className="text-xs text-slate-400">
        Note: this reflects measured, cross-validated baseline model performance, not a claim that the
        DQI score itself predicts accuracy. Metrics are shown as mean ± standard deviation across folds.
        Improving flagged data quality issues is expected to improve real-world performance, but the exact
        relationship is empirical, not guaranteed.
      </p>
    </div>
  )
}

function FeatureSetSection({ title, subtitle, featureSet, metricLabels, highlight }) {
  return (
    <section>
      <div className="flex items-baseline justify-between mb-3">
        <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide">{title}</h3>
        <span className="text-xs text-slate-400">{featureSet.n_features} feature(s)</span>
      </div>
      {subtitle && <p className="text-xs text-slate-500 mb-3">{subtitle}</p>}
      <div className="grid md:grid-cols-2 gap-4">
        {featureSet.results.map((r) => (
          <div
            key={r.model_name}
            className={`bg-white border rounded-xl p-4 ${highlight ? 'border-brand-200 ring-1 ring-brand-100' : 'border-slate-200'}`}
          >
            <h4 className="font-semibold text-sm mb-3">{r.model_name}</h4>
            {r.metrics.error ? (
              <p className="text-sm text-red-600">Training failed: {r.metrics.error}</p>
            ) : (
              <div className="grid grid-cols-3 gap-3">
                {Object.entries(r.metrics).map(([key, stats]) => (
                  <div key={key} className="text-center">
                    <div className="text-lg font-bold text-slate-800">{stats.mean}</div>
                    <div className="text-xs text-slate-400">± {stats.std}</div>
                    <div className="text-xs text-slate-500 mt-0.5">{metricLabels[key] || key}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  )
}

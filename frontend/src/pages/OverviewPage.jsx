import DQIGauge from '../charts/DQIGauge.jsx'
import QualityCard from '../components/QualityCard.jsx'

const EXPLANATIONS = {
  completeness: (a) => a.completeness.explanation,
  consistency: (a) => a.consistency.explanation,
  validity: (a) => a.validity.explanation,
  uniqueness: (a) => a.uniqueness.explanation,
  balance: (a) => a.balance.explanation,
}

export default function OverviewPage({ dataset, analysis }) {
  const { dqi } = analysis

  return (
    <div className="space-y-8">
      <section className="grid md:grid-cols-3 gap-6 items-center bg-white border border-slate-200 rounded-2xl p-6">
        <div className="md:col-span-1 flex justify-center">
          <DQIGauge score={dqi.dqi_0_100} interpretation={dqi.interpretation} />
        </div>
        <div className="md:col-span-2">
          <h2 className="text-lg font-bold mb-1">Dataset Quality Index</h2>
          <p className="text-sm text-slate-500 mb-4">
            {dataset.n_rows.toLocaleString()} rows × {dataset.n_columns} columns · Domain preset: <span className="capitalize font-medium">{dqi.domain_preset}</span>
          </p>
          <p className="text-sm text-slate-600">{dqi.note}</p>
          {analysis.warnings?.length > 0 && (
            <div className="mt-4 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-4 py-3 space-y-1">
              {analysis.warnings.map((w, i) => (
                <div key={i}>⚠ {w}</div>
              ))}
            </div>
          )}
        </div>
      </section>

      <section>
        <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-3">Quality dimensions</h3>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {dqi.components.map((c) => (
            <QualityCard
              key={c.name}
              name={c.name}
              score={c.score ?? 0}
              weight={c.weight}
              contribution={c.contribution}
              available={c.available}
              explanation={EXPLANATIONS[c.name]?.(analysis)}
            />
          ))}
        </div>
      </section>

      <section className="grid sm:grid-cols-3 gap-4">
        <StatCard label="Issues detected" value={analysis.issues.length} />
        <StatCard label="Critical / High severity" value={analysis.issues.filter((i) => ['critical', 'high'].includes(i.severity)).length} />
        <StatCard label="Recommendations" value={analysis.recommendations.length} />
      </section>

      <InformativenessPanel informativeness={analysis.informativeness} />
    </div>
  )
}

function InformativenessPanel({ informativeness }) {
  if (!informativeness.available) return null

  return (
    <section className="bg-violet-50 border border-violet-200 rounded-2xl p-6">
      <div className="flex items-center gap-2 mb-1">
        <h3 className="text-sm font-semibold text-violet-800 uppercase tracking-wide">
          Feature-Target Informativeness
        </h3>
        <span className="text-xs px-2 py-0.5 rounded-full bg-violet-200 text-violet-800">Experimental</span>
      </div>
      <p className="text-xs text-violet-700 mb-4">
        Separate from DQI: this measures whether the data carries predictive signal about the
        target, not whether it's well-formed. A dataset can score high on DQI (clean, complete,
        balanced) while still having little real signal here — they're different properties.
      </p>
      <p className="text-sm text-slate-700 mb-3">{informativeness.explanation}</p>
      <div className="space-y-1.5">
        {informativeness.top_features.slice(0, 5).map((f) => (
          <div key={f.column} className="flex items-center gap-3">
            <span className="text-xs font-mono text-slate-500 w-36 truncate">{f.column}</span>
            <div className="flex-1 h-2 bg-violet-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-violet-500"
                style={{ width: `${Math.min(100, f.normalized_mi * 100)}%` }}
              />
            </div>
            <span className="text-xs text-slate-500 w-12 text-right">{(f.normalized_mi * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </section>
  )
}

function StatCard({ label, value }) {
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 text-center">
      <div className="text-2xl font-bold text-slate-800">{value}</div>
      <div className="text-xs text-slate-500 mt-1">{label}</div>
    </div>
  )
}

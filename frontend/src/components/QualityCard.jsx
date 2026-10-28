export default function QualityCard({ name, score, weight, contribution, available, explanation }) {
  const pct = available ? Math.round(score * 100) : null
  const barColor = !available ? 'bg-slate-300' : pct >= 80 ? 'bg-emerald-500' : pct >= 60 ? 'bg-blue-500' : pct >= 40 ? 'bg-amber-500' : 'bg-red-500'

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4">
      <div className="flex items-baseline justify-between mb-2">
        <h3 className="font-semibold capitalize text-slate-800">{name}</h3>
        <span className="text-lg font-bold text-slate-800">{available ? pct : '—'}</span>
      </div>
      <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden mb-2">
        <div className={`h-full ${barColor}`} style={{ width: `${pct || 0}%` }} />
      </div>
      <p className="text-xs text-slate-500 mb-2">
        Weight: {(weight * 100).toFixed(0)}% · Contribution: {(contribution * 100).toFixed(1)} pts
      </p>
      {explanation && <p className="text-xs text-slate-500">{explanation}</p>}
    </div>
  )
}

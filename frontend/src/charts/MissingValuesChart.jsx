import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'

export default function MissingValuesChart({ perColumnMissingPct }) {
  const data = Object.entries(perColumnMissingPct)
    .map(([column, pct]) => ({ column, pct }))
    .sort((a, b) => b.pct - a.pct)
    .slice(0, 15)

  if (data.every((d) => d.pct === 0)) {
    return <p className="text-sm text-slate-500">No missing values in any column.</p>
  }

  return (
    <ResponsiveContainer width="100%" height={Math.max(200, data.length * 32)}>
      <BarChart data={data} layout="vertical" margin={{ left: 20, right: 20 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" unit="%" domain={[0, 100]} />
        <YAxis type="category" dataKey="column" width={120} tick={{ fontSize: 12 }} />
        <Tooltip formatter={(v) => `${v}%`} />
        <Bar dataKey="pct" fill="#f59e0b" radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

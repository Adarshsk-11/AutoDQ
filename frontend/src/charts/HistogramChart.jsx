import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'

export default function HistogramChart({ histogram }) {
  const { bins, counts } = histogram
  const data = counts.map((count, i) => ({
    range: `${bins[i].toFixed(1)}–${bins[i + 1].toFixed(1)}`,
    count,
  }))

  return (
    <ResponsiveContainer width="100%" height={180}>
      <BarChart data={data} margin={{ left: 0, right: 10, top: 5 }}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="range" tick={{ fontSize: 9 }} interval={Math.ceil(data.length / 8)} />
        <YAxis tick={{ fontSize: 10 }} width={30} />
        <Tooltip />
        <Bar dataKey="count" fill="#3366ff" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

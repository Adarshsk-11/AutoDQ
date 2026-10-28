import { RadialBarChart, RadialBar, PolarAngleAxis } from 'recharts'

const INTERPRETATION_COLORS = {
  Poor: '#dc2626',
  Low: '#f59e0b',
  Moderate: '#3b82f6',
  High: '#16a34a',
}

export default function DQIGauge({ score, interpretation }) {
  const color = INTERPRETATION_COLORS[interpretation] || '#3366ff'
  const data = [{ name: 'DQI', value: score, fill: color }]

  return (
    <div className="flex flex-col items-center">
      <RadialBarChart
        width={220}
        height={160}
        cx={110}
        cy={140}
        innerRadius={80}
        outerRadius={130}
        startAngle={180}
        endAngle={0}
        data={data}
      >
        <PolarAngleAxis type="number" domain={[0, 100]} angleAxisId={0} tick={false} />
        <RadialBar background dataKey="value" cornerRadius={10} angleAxisId={0} />
      </RadialBarChart>
      <div className="-mt-16 text-center">
        <div className="text-4xl font-bold" style={{ color }}>
          {score}
        </div>
        <div className="text-xs text-slate-500">out of 100</div>
      </div>
      <div className="mt-2 text-sm font-semibold px-3 py-1 rounded-full" style={{ backgroundColor: color + '20', color }}>
        {interpretation}
      </div>
    </div>
  )
}

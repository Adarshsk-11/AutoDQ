export default function Spinner({ label }) {
  return (
    <div className="flex items-center gap-2 text-sm text-slate-500">
      <span className="inline-block w-4 h-4 border-2 border-slate-300 border-t-brand-600 rounded-full animate-spin" />
      {label && <span>{label}</span>}
    </div>
  )
}

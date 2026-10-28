export default function Nav({ pages, active, onChange }) {
  return (
    <nav className="bg-white border-b border-slate-200">
      <div className="max-w-6xl mx-auto w-full px-6 flex gap-1 overflow-x-auto">
        {pages.map((page) => (
          <button
            key={page}
            onClick={() => onChange(page)}
            className={`px-4 py-3 text-sm font-medium border-b-2 transition whitespace-nowrap ${
              active === page
                ? 'border-brand-600 text-brand-700'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            {page}
          </button>
        ))}
      </div>
    </nav>
  )
}

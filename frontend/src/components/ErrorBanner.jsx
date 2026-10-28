export default function ErrorBanner({ message, onRetry, retryLabel = 'Try again' }) {
  if (!message) return null
  return (
    <div className="flex items-start justify-between gap-4 text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg px-4 py-3">
      <span>⚠ {message}</span>
      {onRetry && (
        <button
          onClick={onRetry}
          className="shrink-0 text-xs font-medium text-red-700 underline underline-offset-2 hover:no-underline"
        >
          {retryLabel}
        </button>
      )}
    </div>
  )
}

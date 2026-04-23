export default function QueryInput({ value, onChange, onSubmit, loading }) {
  function handleKey(e) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) onSubmit()
  }

  return (
    <div className="query-box">
      <label className="query-label" htmlFor="query">
        Airline support query
      </label>
      <textarea
        id="query"
        className="query-textarea"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKey}
        placeholder="Ask about flights, delays, cancellations, baggage, refunds, check-in, or any airline support issue…"
        rows={3}
        disabled={loading}
      />
      <div className="query-footer">
        <span className="query-hint">Airline &amp; travel support only · Ctrl+Enter to submit</span>
        <button
          className={`query-btn${loading ? ' loading' : ''}`}
          onClick={onSubmit}
          disabled={loading || !value.trim()}
        >
          {loading ? 'Processing…' : 'Ask →'}
        </button>
      </div>
    </div>
  )
}

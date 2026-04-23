function PriorityBadge({ priority }) {
  const cls = priority?.toLowerCase() ?? 'unavailable'
  return <span className={`badge badge--${cls}`}>{priority ?? '—'}</span>
}

function SimBar({ score }) {
  const pct = Math.round((score ?? 0) * 100)
  return (
    <div className="ticket-footer">
      <span className="ticket-sim-label">Similarity</span>
      <div className="sim-bar-wrap">
        <div className="sim-bar" style={{ width: `${pct}%` }} />
      </div>
      <span className="sim-pct">{pct}%</span>
    </div>
  )
}

function TicketCard({ ticket }) {
  const urgent = ticket.priority === 'URGENT'
  const hasResponse = ticket.agent_response && ticket.agent_response.trim().length > 0
  return (
    <div className={`ticket-card${urgent ? ' ticket-card--urgent' : ''}`}>
      <div className="ticket-header">
        <span className="ticket-airline">{ticket.airline}</span>
        <PriorityBadge priority={ticket.priority} />
      </div>
      <p className="ticket-text">{ticket.text}</p>
      {hasResponse && (
        <div className="ticket-agent-response">
          <span className="ticket-agent-label">Agent</span>
          <p className="ticket-agent-text">{ticket.agent_response}</p>
        </div>
      )}
      <SimBar score={ticket.similarity} />
    </div>
  )
}

export default function SourcePanel({ sources, lowSimilarity }) {
  return (
    <section className="section">
      <div className="source-header">
        <h2 className="section-title">
          Retrieved Sources ({sources.length})
        </h2>
        {lowSimilarity && (
          <div className="warning-badge">
            ⚠ Low similarity — RAG context may not be relevant
          </div>
        )}
      </div>

      {sources.length === 0 ? (
        <p className="empty-state">
          No similar tickets found. Check that Qdrant is populated.
        </p>
      ) : (
        <div className="source-grid">
          {sources.map((s, i) => (
            <TicketCard key={i} ticket={s} />
          ))}
        </div>
      )}
    </section>
  )
}

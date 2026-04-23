function fmt(ms) {
  if (ms == null) return '—'
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`
}

function fmtCost(usd) {
  if (usd == null) return '—'
  if (usd === 0) return '$0.00'
  return `$${usd.toFixed(6)}`
}

function AnswerCard({ title, answer, latencyMs, costUsd, variant }) {
  return (
    <div className={`answer-card answer-card--${variant}`}>
      <div className="answer-card-header">
        <span className="answer-card-title">{title}</span>
        <div className="answer-card-meta">
          <span>⏱ {fmt(latencyMs)}</span>
          <span>{fmtCost(costUsd)}</span>
        </div>
      </div>
      <p className="answer-card-text">{answer}</p>
    </div>
  )
}

export default function AnswerSection({ result }) {
  const { rag_answer, non_rag_answer, metrics } = result
  return (
    <section className="section">
      <h2 className="section-title">Generated Answers</h2>
      <div className="answer-grid">
        <AnswerCard
          title="RAG Answer — with retrieved context"
          answer={rag_answer}
          latencyMs={metrics?.rag_ms}
          costUsd={metrics?.rag_cost_usd}
          variant="rag"
        />
        <AnswerCard
          title="Non-RAG Answer — LLM alone"
          answer={non_rag_answer}
          latencyMs={metrics?.non_rag_ms}
          costUsd={metrics?.non_rag_cost_usd}
          variant="plain"
        />
      </div>
    </section>
  )
}

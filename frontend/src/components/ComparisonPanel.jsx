function fmt(n, dec = 2) {
  return n != null ? Number(n).toFixed(dec) : '—'
}

function fmtCost(usd) {
  if (usd == null) return '—'
  if (usd === 0) return '$0.000000'
  return `$${Number(usd).toFixed(6)}`
}

function fmtPct(ratio) {
  return ratio != null ? `${(ratio * 100).toFixed(1)}%` : '—'
}

function PriorityBadge({ label }) {
  const cls = label?.toLowerCase() ?? 'unavailable'
  return <span className={`badge badge--${cls}`}>{label ?? '—'}</span>
}

function Row({ label, ml, llm, mlClass = '', llmClass = '' }) {
  return (
    <tr>
      <td>{label}</td>
      <td className={mlClass}>{ml}</td>
      <td className={llmClass}>{llm}</td>
    </tr>
  )
}

export default function ComparisonPanel({ result, trainedMetrics }) {
  const { ml_prediction: ml, llm_prediction: llm, metrics } = result

  const bestKey = trainedMetrics?.best_model
  const bestDisplay = bestKey?.replace(/_/g, ' ') ?? ''
  const bestM = trainedMetrics?.models?.[bestKey]

  // Cost at scale: 10,000 tickets per hour
  const llmCost10k = llm?.cost_usd != null
    ? `$${(llm.cost_usd * 10_000).toFixed(2)}/hr`
    : '—'

  return (
    <section className="section">
      <h2 className="section-title">Priority Prediction — Comparison</h2>

      {/* ── Answer latency/cost sub-header ── */}
      <div className="answer-meta-row">
        <div className="answer-meta-card">
          <h4>RAG Answer</h4>
          <div className="stat">
            <span>⏱ {fmt(metrics?.rag_ms, 0)} ms</span>
            <span>💰 {fmtCost(metrics?.rag_cost_usd)}</span>
          </div>
        </div>
        <div className="answer-meta-card">
          <h4>Non-RAG Answer</h4>
          <div className="stat">
            <span>⏱ {fmt(metrics?.non_rag_ms, 0)} ms</span>
            <span>💰 {fmtCost(metrics?.non_rag_cost_usd)}</span>
          </div>
        </div>
      </div>

      {/* ── Main comparison table ── */}
      <div className="comparison-wrap">
        <table className="comparison-table">
          <thead>
            <tr>
              <th />
              <th className="col-ml">
                ML Model
                {bestDisplay && (
                  <span className="model-subname">{bestDisplay}</span>
                )}
              </th>
              <th className="col-llm">LLM Zero-Shot</th>
            </tr>
          </thead>
          <tbody>
            <Row
              label="Prediction"
              ml={<PriorityBadge label={ml?.label} />}
              llm={<PriorityBadge label={llm?.label} />}
            />
            <Row
              label="Confidence"
              ml={ml?.confidence != null ? fmtPct(ml.confidence) : '—'}
              llm={<span className="dim">—</span>}
              llmClass="dim"
            />
            <Row
              label="Test Accuracy"
              ml={bestM ? fmtPct(bestM.test_accuracy ?? bestM.accuracy) : <span className="dim">run training</span>}
              llm={<span className="dim">—</span>}
              llmClass="dim"
            />
            <Row
              label="F1 (test)"
              ml={bestM ? fmt(bestM.test_f1 ?? bestM.f1, 3) : '—'}
              llm={<span className="dim">—</span>}
              llmClass="dim"
            />
            <Row
              label="ROC-AUC (test)"
              ml={bestM ? fmt(bestM.test_roc_auc ?? bestM.roc_auc, 3) : '—'}
              llm={<span className="dim">—</span>}
              llmClass="dim"
            />
            <Row
              label="Latency / call"
              ml={
                ml?.latency_ms != null
                  ? <strong>{fmt(ml.latency_ms, 3)} ms</strong>
                  : '—'
              }
              mlClass="win"
              llm={llm?.latency_ms != null ? `${fmt(llm.latency_ms, 0)} ms` : '—'}
            />
            <Row
              label="Cost / call"
              ml={<strong>$0.000000</strong>}
              mlClass="cost-zero"
              llm={fmtCost(llm?.cost_usd)}
            />
            <Row
              label="Cost @ 10,000 tickets / hr"
              ml={<strong>$0.00</strong>}
              mlClass="cost-zero"
              llm={llmCost10k}
            />
          </tbody>
        </table>
      </div>

      {/* ── Inline production recommendation ── */}
      <div className="recommendation">
        <h3>Production Recommendation</h3>
        <p>
          At 10,000 tickets/hour the ML model costs{' '}
          <strong>$0.00</strong> and runs in{' '}
          <strong>
            {ml?.latency_ms != null ? `${fmt(ml.latency_ms, 3)} ms` : 'sub-millisecond time'}
          </strong>
          . The LLM zero-shot costs{' '}
          <strong>{llmCost10k}</strong> and takes ~
          <strong>
            {llm?.latency_ms != null ? `${fmt(llm.latency_ms, 0)} ms` : '~500 ms'}
          </strong>{' '}
          per call. Deploy the ML classifier for high-volume triage. Reserve
          the LLM for low-volume edge cases that need natural-language explanation
          or where accuracy on novel phrasing is critical.
        </p>
      </div>
    </section>
  )
}

const EXAMPLES = [
  'My flight was cancelled — how do I rebook?',
  'My bag didn\'t arrive at the destination.',
  'I\'ve been waiting 4 hours for a delayed flight, what are my options?',
  'How do I request a refund for a cancelled booking?',
  'I was charged twice for my seat upgrade.',
]

const REASON_LABELS = {
  code_request:          'Code writing is outside this assistant\'s scope.',
  arithmetic_expression: 'Maths and calculations are outside this assistant\'s scope.',
  creative_writing:      'Creative writing is outside this assistant\'s scope.',
  general_trivia:        'General knowledge questions are outside this assistant\'s scope.',
  jailbreak_attempt:     'That type of instruction is not permitted here.',
  off_topic:             'Your question doesn\'t appear to be about airline customer support.',
}

export default function OffTopicBanner({ info, onDismiss }) {
  if (!info) return null

  const reasonLabel = REASON_LABELS[info.reason] ?? info.message

  return (
    <div className="off-topic-banner" role="alert">
      <div className="off-topic-header">
        <span className="off-topic-icon">⚠</span>
        <div className="off-topic-titles">
          <strong className="off-topic-title">Out of scope</strong>
          <span className="off-topic-reason">{reasonLabel}</span>
        </div>
        <button className="off-topic-dismiss" onClick={onDismiss} aria-label="Dismiss">✕</button>
      </div>

      <p className="off-topic-hint">{info.hint}</p>

      <div className="off-topic-examples">
        <span className="off-topic-examples-label">Try questions like:</span>
        <ul className="off-topic-list">
          {EXAMPLES.map((ex, i) => (
            <li key={i}>{ex}</li>
          ))}
        </ul>
      </div>
    </div>
  )
}

import Tooltip from '../components/Tooltip'

const FEATURES = [
  { name: 'text_length',       desc: 'Total character count. Urgent messages tend to be longer — more detail = more urgency.' },
  { name: 'word_count',        desc: 'Number of words. Correlated with complexity and emotional detail.' },
  { name: 'exclamation_count', desc: 'Number of ! marks. Signals frustration or emotional urgency.' },
  { name: 'question_count',    desc: 'Number of ? marks. High counts suggest confusion or demands for answers.' },
  { name: 'caps_ratio',        desc: 'Fraction of letters that are uppercase. SHOUTING is a strong urgency signal.' },
  { name: 'has_refund',        desc: 'Contains the word "refund". Financial disputes often need escalation.' },
  { name: 'has_cancel',        desc: 'Contains cancel/cancelled. Flight cancellations are frequently urgent.' },
  { name: 'has_delay',         desc: 'Delay-related terms (delayed, late, waiting, on hold…). The strongest single predictor.' },
  { name: 'has_help',          desc: 'Contains "help". Direct request for assistance.' },
  { name: 'has_broken',        desc: 'Contains "broken". Equipment or service failure.' },
  { name: 'has_stranded',      desc: 'Contains "stranded". One of the hard-override critical keywords.' },
  { name: 'has_medical',       desc: 'Contains "medical". Any health-related mention triggers max urgency.' },
  { name: 'profanity_count',   desc: 'Count of strong negative words (worst, awful, terrible…). High emotional distress.' },
  { name: 'has_time_mention',  desc: 'Pattern like "3 hours" or "2 days". Quantified waits are concrete urgency evidence.' },
]

function MetricBar({ value, max = 1, color = 'blue' }) {
  if (value == null) return <span className="metric-bar-val dim">—</span>
  const pct = Math.round((value / max) * 100)
  return (
    <div className="metric-bar-wrap">
      <div className={`metric-bar metric-bar--${color}`} style={{ width: `${pct}%` }} />
      <span className="metric-bar-val">{value.toFixed(3)}</span>
    </div>
  )
}

function SplitBadge({ label, size, color }) {
  return (
    <div className={`split-badge split-badge--${color}`}>
      <span className="split-badge-label">{label}</span>
      <span className="split-badge-size">{size?.toLocaleString() ?? '—'}</span>
    </div>
  )
}

export default function ModelPage({ trainedMetrics }) {
  const hasMetrics = !!trainedMetrics

  const best_model        = trainedMetrics?.best_model
  const models            = trainedMetrics?.models
  const dataset_size      = trainedMetrics?.dataset_size
  const train_size        = trainedMetrics?.train_size
  const val_size          = trainedMetrics?.val_size
  const test_size         = trainedMetrics?.test_size
  const class_distribution = trainedMetrics?.class_distribution
  const features          = trainedMetrics?.features ?? FEATURES.map(f => f.name)
  const trained_at        = trainedMetrics?.trained_at

  const bestDisplay = best_model?.replace(/_/g, ' ')
  const bestM       = models?.[best_model]

  const urgentCount = class_distribution?.['1'] ?? 0
  const normalCount = class_distribution?.['0'] ?? 0
  const urgentPct   = dataset_size ? Math.round((urgentCount / dataset_size) * 100) : 18

  return (
    <div className="page">
      <div className="page-hero">
        <h2 className="page-title">Model & Metrics</h2>
        <p className="page-lead">
          Three classifiers trained on a 60 / 20 / 20 split. The validation set selects the
          best model; test set metrics are the final, untouched evaluation.
          {trained_at && (
            <> Trained{' '}
              {new Date(trained_at).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })}.
            </>
          )}
        </p>
      </div>

      {/* ── Not trained notice ──────────────────────────────────────────── */}
      {!hasMetrics && (
        <div className="info-banner">
          <strong>Metrics not available.</strong> Run{' '}
          <code>uv run python scripts/train_models.py</code> in the backend directory to generate them.
          The features, approach, and caveat sections below are always shown.
        </div>
      )}

      {/* ── Split breakdown ─────────────────────────────────────────────── */}
      <div className="split-row">
        <SplitBadge label="Train"      size={train_size}  color="blue" />
        <SplitBadge label="Validation" size={val_size}    color="purple" />
        <SplitBadge label="Test"       size={test_size}   color="green" />
        <div className="split-note">
          Model selected by <strong>val F1</strong> · final numbers from <strong>test set</strong>
        </div>
      </div>

      {/* ── Best model callout ──────────────────────────────────────────── */}
      {hasMetrics && (
        <section className="section best-model-card">
          <div className="best-model-inner">
            <div>
              <div className="best-model-label">Best Model (test set)</div>
              <div className="best-model-name">{bestDisplay}</div>
            </div>
            <div className="best-model-stats">
              <div className="bm-stat">
                <span className="bm-stat-val">{bestM?.test_f1?.toFixed(3) ?? '—'}</span>
                <span className="bm-stat-label">
                  F1 Score{' '}
                  <Tooltip text="Balances precision (don't false-alarm) and recall (don't miss urgent tickets). Better than accuracy on imbalanced data." />
                </span>
              </div>
              <div className="bm-stat">
                <span className="bm-stat-val">{bestM?.test_accuracy?.toFixed(3) ?? '—'}</span>
                <span className="bm-stat-label">
                  Accuracy{' '}
                  <Tooltip text="Fraction of all predictions correct. Can be misleading on imbalanced classes — see caveat." />
                </span>
              </div>
              <div className="bm-stat">
                <span className="bm-stat-val">{bestM?.test_roc_auc?.toFixed(3) ?? '—'}</span>
                <span className="bm-stat-label">
                  ROC-AUC{' '}
                  <Tooltip text="Probability the model ranks an URGENT ticket above a NORMAL one. 0.5 = random, 1.0 = perfect." />
                </span>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* ── All-models comparison table ─────────────────────────────────── */}
      {hasMetrics && models && (
        <section className="section">
          <h3 className="section-title">All Three Models — Validation vs Test</h3>
          <p className="section-sub">
            Validation metrics were used to pick the winner. Test metrics are the
            final honest evaluation — each model only touched the test set once.
            Best model highlighted; selected by highest validation F1.
          </p>
          <div className="comparison-wrap">
            <table className="comparison-table model-comparison">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>
                    Val F1{' '}
                    <Tooltip text="Used for model selection. Higher = better at catching URGENT without too many false alarms." />
                  </th>
                  <th>Val Accuracy</th>
                  <th>Val AUC</th>
                  <th>
                    Test F1{' '}
                    <Tooltip text="Final report. Evaluated once after the winner was chosen." />
                  </th>
                  <th>Test Accuracy</th>
                  <th>Test AUC</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(models).map(([name, m]) => (
                  <tr key={name} className={name === best_model ? 'row-best' : ''}>
                    <td>
                      {name.replace(/_/g, ' ')}
                      {name === best_model && <span className="winner-badge">deployed</span>}
                    </td>
                    <td><MetricBar value={m.val_f1}        color="purple" /></td>
                    <td><MetricBar value={m.val_accuracy}  color="blue"   /></td>
                    <td><MetricBar value={m.val_roc_auc}   color="purple" /></td>
                    <td><MetricBar value={m.test_f1}       color="green"  /></td>
                    <td><MetricBar value={m.test_accuracy} color="blue"   /></td>
                    <td><MetricBar value={m.test_roc_auc}  color="green"  /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* ── Class distribution ─────────────────────────────────────────────── */}
      <section className="section">
        <h3 className="section-title">Class Distribution</h3>
        <p className="section-sub">
          Imbalanced datasets are why accuracy is a misleading metric — a model that
          always guesses NORMAL would score {100 - urgentPct}% accuracy while being completely useless.
          F1 is the primary selection criterion for exactly this reason.
        </p>
        <div className="class-bars">
          <div className="class-bar-row">
            <span className="class-bar-label">NORMAL</span>
            <div className="class-bar-track">
              <div className="class-bar class-bar--normal" style={{ width: `${100 - urgentPct}%` }} />
            </div>
            <span className="class-bar-count">
              {hasMetrics ? `${normalCount.toLocaleString()} (${100 - urgentPct}%)` : `~${100 - urgentPct}% (estimated)`}
            </span>
          </div>
          <div className="class-bar-row">
            <span className="class-bar-label">URGENT</span>
            <div className="class-bar-track">
              <div className="class-bar class-bar--urgent" style={{ width: `${urgentPct}%` }} />
            </div>
            <span className="class-bar-count">
              {hasMetrics ? `${urgentCount.toLocaleString()} (${urgentPct}%)` : `~${urgentPct}% (estimated)`}
            </span>
          </div>
        </div>
      </section>

      {/* ── Features ───────────────────────────────────────────────────────── */}
      <section className="section">
        <h3 className="section-title">Features Used ({FEATURES.length})</h3>
        <p className="section-sub">
          These 14 features are extracted from raw text in milliseconds and are independent
          of the labeling rule — the labeler uses broad boolean flags, while these features
          provide more granular signal (counts, ratios, specific keyword flags).
        </p>
        <div className="feature-grid">
          {FEATURES.map(f => (
            <div key={f.name} className="feature-row">
              <code className="feature-name">{f.name}</code>
              <span className="feature-desc">{f.desc}</span>
            </div>
          ))}
        </div>
      </section>

      {/* ── Honest caveat ──────────────────────────────────────────────────── */}
      <div className="caveat-box caveat-box--wide">
        <strong className="caveat-title">What does ~92% test accuracy actually mean here?</strong>
        <p>
          The test set accuracy measures how well the model reproduces our <em>labeling rule</em> on
          tweets it has never seen — not how well it matches real-world expert judgment.
          Our labels (URGENT / NORMAL) were created by a regex keyword-scoring algorithm, not by
          human annotators. There is no ground-truth record of how airlines actually handled each case.
        </p>
        <p>
          In practice: the model is very good at applying our rule consistently at scale.
          Whether that rule correctly identifies truly urgent situations is a separate question
          that would require human annotation to answer.
        </p>
      </div>
    </div>
  )
}

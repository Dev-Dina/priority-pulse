import Tooltip from '../components/Tooltip'

const SELECTED_AIRLINES = [
  { name: 'American Airlines', handle: '@americanair' },
  { name: 'Delta',             handle: '@delta' },
  { name: 'Southwest',         handle: '@southwestair' },
  { name: 'JetBlue',           handle: '@jetblue' },
  { name: 'United',            handle: '@united' },
  { name: 'US Airways',        handle: '@usairways' },
]

const LABELING_STEPS = [
  {
    num: '1',
    title: 'Hard override — critical keywords',
    detail: 'If the tweet contains any critical keyword (stranded, medical, police, sue, fraud, disaster…) → immediately URGENT with score 10. These represent situations where a missed flag could be a safety or legal issue.',
  },
  {
    num: '2',
    title: 'Additive scoring',
    detail: 'Delay signal (+3), urgency keywords (+2), profanity/negative words (+1), all-caps ratio > 40% (+1), 2+ exclamation marks (+1). Points accumulate.',
  },
  {
    num: '3',
    title: 'Threshold — score ≥ 3 → URGENT',
    detail: 'Final classification. Score ≥ 3 is URGENT, below is NORMAL. The raw score (0–10) is stored so the UI can show severity within URGENT tickets.',
  },
]

function RuleRow({ keyword, pattern, reason }) {
  return (
    <tr>
      <td><code>{keyword}</code></td>
      <td className="dim">{pattern}</td>
      <td>{reason}</td>
    </tr>
  )
}

function FunnelRow({ label, value, sub, highlight }) {
  return (
    <div className={`funnel-row${highlight ? ' funnel-row--highlight' : ''}`}>
      <span className="funnel-label">{label}</span>
      <span className="funnel-value">{value}</span>
      {sub && <span className="funnel-sub">{sub}</span>}
    </div>
  )
}

function RemovalRow({ label, value, pct }) {
  return (
    <div className="removal-row">
      <span className="removal-label">{label}</span>
      <span className="removal-value">−{value?.toLocaleString() ?? '—'}</span>
      {pct != null && <span className="removal-pct">{pct}%</span>}
    </div>
  )
}

export default function DataPage({ dataStats }) {
  const s = dataStats

  // Derive display values — fall back to known-correct hardcoded values if
  // pipeline hasn't been run yet so the page is never blank.
  //
  // raw_total     = all rows in twcs.csv (inbound + outbound combined, ~2.8M)
  // raw_inbound   = inbound-only tweets before airline filter (~half of raw_total)
  // after_filter  = inbound tweets for the 6 target airlines (main selection step)
  // final_count   = after quality cleaning (empty, low-quality, duplicates removed)
  const rawTotal       = s?.raw_total             ?? 2_811_774
  const rawInbound     = s?.raw_inbound_count      ?? 1_382_038
  const afterFilter    = s?.after_airline_filter   ?? 91_744
  const removedEmpty   = s?.removed_empty          ?? 630
  const removedLow     = s?.removed_low_quality    ?? 2_260
  const removedDupes   = s?.removed_duplicates     ?? 1_006
  const finalCount     = s?.final_count            ?? 87_848
  const withResponse   = s?.with_agent_response    ?? 64_919
  const responsePct    = s?.agent_response_pct     ?? 73.9
  const urgentCount    = s?.labeling?.urgent       ?? 16_235
  const normalCount    = s?.labeling?.normal       ?? 71_613
  const urgentPct      = s?.labeling?.urgent_pct   ?? 18.5
  const otherHandles   = s?.other_top_handles      ?? []
  const ranAt          = s?.pipeline_run_at

  // Quality-filtering removals only (after airline filter — not the main drop)
  const totalRemoved = removedEmpty + removedLow + removedDupes

  // Dynamic stat card label for raw dataset size
  const rawDisplay = rawTotal >= 1_000_000
    ? `~${(rawTotal / 1_000_000).toFixed(1)}M`
    : rawTotal.toLocaleString()

  return (
    <div className="page">
      <div className="page-hero">
        <h2 className="page-title">Data & Labeling</h2>
        <p className="page-lead">
          Where the data comes from, how it was filtered, and how URGENT / NORMAL labels
          were created — including the honest limitations of that process.
        </p>
        {ranAt && (
          <p className="page-run-at">
            Pipeline last run: {new Date(ranAt).toLocaleString('en-GB', {
              day: 'numeric', month: 'short', year: 'numeric',
              hour: '2-digit', minute: '2-digit',
            })}
          </p>
        )}
      </div>

      {/* ── Top stats ─────────────────────────────────────────────────── */}
      <div className="info-grid info-grid--3">
        <div className="stat-card">
          <div className="stat-card-num">{rawDisplay}</div>
          <div className="stat-card-label">Raw tweets in TWCS</div>
          <div className="stat-card-sub">All inbound + outbound tweets in CSV</div>
        </div>
        <div className="stat-card">
          <div className="stat-card-num">{finalCount.toLocaleString()}</div>
          <div className="stat-card-label">Clean tickets selected</div>
          <div className="stat-card-sub">Inbound customer tweets, 6 airlines</div>
        </div>
        <div className="stat-card">
          <div className="stat-card-num">{urgentPct}%</div>
          <div className="stat-card-label">Labelled URGENT</div>
          <div className="stat-card-sub">{urgentCount.toLocaleString()} urgent · {normalCount.toLocaleString()} normal</div>
        </div>
      </div>

      {/* ── Cleaning funnel ────────────────────────────────────────────── */}
      <section className="section">
        <h3 className="section-title">
          Cleaning Funnel{' '}
          <Tooltip text="Shows exactly how many records were dropped at each step so the pipeline is auditable end-to-end." />
        </h3>
        <p className="section-sub">
          The {rawTotal.toLocaleString()} raw tweets include both inbound (customer) and
          outbound (agent) messages across many companies and industries. Filtering to
          inbound-only tweets for our 6 target airlines gives {afterFilter.toLocaleString()} candidate
          tickets. A further {totalRemoved.toLocaleString()} were removed in quality cleaning,
          leaving {finalCount.toLocaleString()} final tickets.
          Of those, {responsePct}% have a matched agent response attached for RAG context.
        </p>

        <div className="funnel-wrap">
          <FunnelRow
            label="Raw dataset (all tweets — inbound + outbound)"
            value={rawTotal.toLocaleString()}
            highlight
          />
          <div className="funnel-arrow">↓ filter to inbound customer tweets only</div>
          <FunnelRow
            label="Inbound tweets (all companies)"
            value={rawInbound.toLocaleString()}
            sub={`−${(rawTotal - rawInbound).toLocaleString()} outbound agent tweets set aside`}
          />
          <div className="funnel-arrow">↓ match one of 6 airline @handles</div>
          <FunnelRow
            label="After airline filter (6 airlines)"
            value={afterFilter.toLocaleString()}
            sub={`−${(rawInbound - afterFilter).toLocaleString()} non-airline mentions removed`}
          />
          <div className="funnel-removals">
            <RemovalRow label="Empty after text cleaning" value={removedEmpty} />
            <RemovalRow label="Low-quality (too short, bare 'thanks'…)" value={removedLow} />
            <RemovalRow label="Duplicates (same cleaned text)" value={removedDupes} />
          </div>
          <div className="funnel-arrow">↓</div>
          <FunnelRow
            label="Final clean dataset"
            value={finalCount.toLocaleString()}
            highlight
          />
          <div className="funnel-agent-row">
            <span className="agent-label">Agent response joined</span>
            <span className="agent-value">{withResponse.toLocaleString()}</span>
            <span className="agent-pct">{responsePct}% of tickets</span>
            <Tooltip text="For each customer tweet, we look up the support agent's reply from outbound tweets and attach it. This enriches the RAG chunk with both the problem and the resolution." />
          </div>
        </div>
      </section>

      {/* ── Source dataset ───────────────────────────────────────────────── */}
      <section className="section">
        <h3 className="section-title">Selected Airlines</h3>
        <p className="section-sub">
          The <em>Twitter Customer Support Corpus</em> (Kaggle) contains real Twitter interactions
          collected circa 2017–2018. We filter to 6 major US airline handles.
        </p>
        <div className="airline-grid">
          {SELECTED_AIRLINES.map(a => (
            <div key={a.handle} className="airline-chip">
              <span className="airline-chip-handle">{a.handle}</span>
              <span className="airline-chip-name">{a.name}</span>
            </div>
          ))}
        </div>
      </section>

      {/* ── Other brands in TWCS ─────────────────────────────────────────── */}
      {otherHandles.length > 0 && (
        <section className="section">
          <h3 className="section-title">
            Other Brands in TWCS — Not Selected{' '}
            <Tooltip text="TWCS is not airline-only. These are the top other companies by tweet volume that we deliberately excluded to keep the domain focused." />
          </h3>
          <p className="section-sub">
            TWCS spans many industries. The handles below appeared at least 200 times
            but were excluded because this project focuses on airline support specifically.
          </p>
          <div className="other-handles-grid">
            {otherHandles.map(h => (
              <div key={h.handle} className="other-chip">
                <span className="other-chip-handle">@{h.handle}</span>
                <span className="other-chip-count">{h.count.toLocaleString()} tweets</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ── Filtering steps ──────────────────────────────────────────────── */}
      <section className="section">
        <h3 className="section-title">
          Filtering — What We Keep and Why{' '}
          <Tooltip text="Inbound = customer tweeting at the airline. Outbound = airline replying. We keep only inbound for classification, but JOIN the outbound response for RAG context." />
        </h3>
        <div className="filter-steps">
          <div className="filter-step">
            <div className="filter-step-icon filter-step-icon--keep">keep</div>
            <div>
              <strong>Inbound tweets (inbound = True)</strong>
              <p>Customer-to-airline messages. The complaint, problem, or question. This is what we classify.</p>
            </div>
          </div>
          <div className="filter-step">
            <div className="filter-step-icon filter-step-icon--join">join</div>
            <div>
              <strong>Outbound tweets (inbound = False) — joined, not dropped</strong>
              <p>
                Airline replies. We look up each customer tweet's response via{' '}
                <code>in_response_to_tweet_id</code> and attach the cleaned agent text as{' '}
                <code>agent_response</code>. This gives RAG both the problem and the resolution.
                The response is stored in the vector payload and shown in the source panel.
              </p>
            </div>
          </div>
          <div className="filter-step">
            <div className="filter-step-icon filter-step-icon--drop">drop</div>
            <div>
              <strong>Non-airline mentions</strong>
              <p>Only tweets whose first @mention is one of the 6 target airline handles are kept.</p>
            </div>
          </div>
          <div className="filter-step">
            <div className="filter-step-icon filter-step-icon--drop">drop</div>
            <div>
              <strong>Low-quality content</strong>
              <p>Tweets under 8 characters, bare "DM sent", "thanks", "help" are removed — they carry no classifiable signal.</p>
            </div>
          </div>
        </div>
      </section>

      {/* ── Text cleaning ─────────────────────────────────────────────────── */}
      <section className="section">
        <h3 className="section-title">Text Cleaning</h3>
        <div className="clean-steps">
          <div className="clean-step"><span className="clean-badge">1</span>Remove URLs — <code>http://...</code> links add no signal</div>
          <div className="clean-step"><span className="clean-badge">2</span>Remove leading @mention — <code>@Delta</code> at the start is noise (airline tracked separately)</div>
          <div className="clean-step"><span className="clean-badge">3</span>Decode HTML entities — <code>&amp;amp;</code> → <code>&amp;</code></div>
          <div className="clean-step"><span className="clean-badge">4</span>Collapse whitespace</div>
          <div className="clean-step"><span className="clean-badge">5</span>Deduplicate on cleaned text — <strong>{removedDupes.toLocaleString()}</strong> duplicates removed</div>
        </div>
        <p className="section-note">
          The original uncleaned tweet is preserved in <code>original_text</code> and shown in the source panel.
          The cleaned version feeds both ML features and vector embeddings.
        </p>
      </section>

      {/* ── Labeling ─────────────────────────────────────────────────────── */}
      <section className="section">
        <h3 className="section-title">
          Labeling — SLA-Based Weak Supervision{' '}
          <Tooltip text="Weak supervision: labels are generated by a rule/heuristic, not a human. Scales to 87k tickets but is less accurate than expert annotation." />
        </h3>
        <p className="section-sub">
          No human annotator labelled these {finalCount.toLocaleString()} tweets. A scoring rule was designed
          to approximate how an SLA-based triage system would prioritise incoming tickets.
        </p>
        <div className="label-steps">
          {LABELING_STEPS.map(s => (
            <div key={s.num} className="label-step">
              <div className="label-step-num">{s.num}</div>
              <div>
                <strong>{s.title}</strong>
                <p>{s.detail}</p>
              </div>
            </div>
          ))}
        </div>

        <h4 className="subsection-title" style={{ marginTop: '1.5rem' }}>Critical keywords (hard override → URGENT immediately)</h4>
        <div className="comparison-wrap">
          <table className="comparison-table">
            <thead>
              <tr><th>Keyword</th><th>Pattern</th><th>Why it overrides</th></tr>
            </thead>
            <tbody>
              <RuleRow keyword="stranded"          pattern="\bstranded\b"   reason="Passenger physically unable to leave airport — safety risk" />
              <RuleRow keyword="medical"           pattern="\bmedical\b"    reason="Any health mention is treated as potential emergency" />
              <RuleRow keyword="sue / lawsuit"     pattern="\bsue\b · \blawsuit\b" reason="Legal threat — requires legal/compliance team" />
              <RuleRow keyword="fraud"             pattern="\bfraud\b"      reason="Financial crime allegation — regulatory obligation to escalate" />
              <RuleRow keyword="unacceptable / disaster" pattern="regex"   reason="Strong complaint language indicating service breakdown" />
            </tbody>
          </table>
        </div>
      </section>

      {/* ── Honest limitations ───────────────────────────────────────────── */}
      <section className="section">
        <h3 className="section-title">Honest Limitations</h3>
        <div className="caveat-grid">
          <div className="caveat-box">
            <strong className="caveat-title">No ground truth</strong>
            <p>
              We have no record of how the airline actually handled each tweet. Our URGENT label
              means "this tweet matches our urgency rule", not "this was treated as urgent in practice."
            </p>
          </div>
          <div className="caveat-box">
            <strong className="caveat-title">Keyword bias</strong>
            <p>
              A tweet saying "my flight is wonderful — cancelled at the last second" would score
              differently depending on which keyword fires first. Irony and context are not
              captured by regex.
            </p>
          </div>
          <div className="caveat-box">
            <strong className="caveat-title">2017–2018 data</strong>
            <p>
              The dataset is ~7 years old. Language patterns, airline policies, and Twitter
              culture have all changed. A model trained here may not generalise to modern tweets.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}

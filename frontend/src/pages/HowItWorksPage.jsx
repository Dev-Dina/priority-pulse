import Tooltip from '../components/Tooltip'

function Step({ num, title, sub, color }) {
  return (
    <div className={`pipe-step pipe-step--${color}`}>
      <div className="pipe-num">{num}</div>
      <div className="pipe-body">
        <strong>{title}</strong>
        <span>{sub}</span>
      </div>
    </div>
  )
}

function Branch({ label, detail, color }) {
  return (
    <div className={`pipe-branch pipe-branch--${color}`}>
      <strong>{label}</strong>
      <p>{detail}</p>
    </div>
  )
}

function Card({ title, children }) {
  return (
    <div className="info-card">
      <h3 className="info-card-title">{title}</h3>
      <div className="info-card-body">{children}</div>
    </div>
  )
}

function Caveat({ title, children }) {
  return (
    <div className="caveat-box">
      <strong className="caveat-title">{title}</strong>
      <p>{children}</p>
    </div>
  )
}

export default function HowItWorksPage() {
  return (
    <div className="page">
      <div className="page-hero">
        <h2 className="page-title">How It Works</h2>
        <p className="page-lead">
          PriorityPulse is a Decision Intelligence system. Every query runs four
          parallel processes — two for generating answers, two for classifying urgency —
          then displays all their outputs side-by-side with latency and cost.
        </p>
      </div>

      {/* ── Pipeline diagram ──────────────────────────────────────────────── */}
      <section className="section">
        <h3 className="section-title">Request Pipeline</h3>
        <div className="pipeline">
          <Step num="1" color="slate" title="Your query arrives"
            sub="A free-text customer support message — e.g. 'My flight was cancelled and I am stranded.'" />
          <div className="pipe-arrow">↓</div>
          <Step num="2" color="blue"
            title="Embedding: text → 384 numbers"
            sub="all-MiniLM-L6-v2 (sentence-transformers) encodes your text into a 384-dimensional vector. Each dimension captures a semantic aspect. Similar sentences land near each other in this space." />
          <div className="pipe-arrow">↓</div>
          <Step num="3" color="purple"
            title="Qdrant: semantic search over 87,848 past tweets"
            sub="The 384-dim vector is compared to every stored tweet embedding using cosine similarity. The top-5 most semantically similar past tickets are returned — even if they use completely different words." />
          <div className="pipe-arrow">↓  four things happen in parallel</div>
          <div className="pipe-branches">
            <Branch color="rag" label="RAG Answer"
              detail="LLM receives your query + the 5 retrieved tweets as context. Grounded in real historical cases." />
            <Branch color="plain" label="Plain Answer"
              detail="LLM receives only your query. No context. Shows what the model knows without historical data." />
            <Branch color="ml" label="ML Classifier"
              detail="14 text features extracted instantly. Gradient Boosting predicts URGENT or NORMAL in ~1 ms at zero cost." />
            <Branch color="llm0" label="LLM Zero-Shot"
              detail="LLM asked directly: 'Is this URGENT or NORMAL?' No examples, no training, just the question." />
          </div>
        </div>
      </section>

      {/* ── Concept cards ─────────────────────────────────────────────────── */}
      <div className="info-grid">

        <Card title={<>What is RAG? <Tooltip text="Retrieval-Augmented Generation — the LLM answer is 'augmented' by retrieved real examples before generating a response." /></>}>
          <p>
            A plain LLM answers from its training data — knowledge frozen at a cutoff date,
            with no access to your specific airline's cases or policies.
          </p>
          <p>
            RAG adds a retrieval step: before asking the LLM, we find the 5 most similar
            past support tickets and inject them as context. The LLM then generates an answer
            that is grounded in actual historical cases rather than guessing.
          </p>
          <p>
            The comparison panel shows whether the RAG answer is more relevant, more specific,
            or more confident than the plain answer on your query.
          </p>
        </Card>

        <Card title={<>Why a Vector Database? <Tooltip text="Qdrant stores tweets as 384-dim vectors. Similarity search finds semantically related tweets, not just keyword matches." /></>}>
          <p>
            A keyword search for "stranded at airport" would miss "stuck at the gate with no
            crew" entirely — same urgency, zero word overlap.
          </p>
          <p>
            Qdrant stores every tweet as a point in 384-dimensional space. Nearby points mean
            nearby meaning. A cosine similarity search finds semantically close tickets
            regardless of the exact words used.
          </p>
          <p>
            87,848 tweet vectors are stored. A similarity threshold (0.3) filters out
            low-relevance matches — when no good match exists, the UI shows a warning banner.
          </p>
        </Card>

        <Card title={<>ML Classifier vs LLM Zero-Shot <Tooltip text="The ML model was trained on our labeled dataset. LLM zero-shot uses the LLM with no examples — just the question." /></>}>
          <p>
            <strong>ML Classifier (Gradient Boosting):</strong> trained on 87,848 labeled tweets.
            Extracts 14 handcrafted features (keyword flags, text stats) and predicts in ~1 ms
            at zero marginal cost. Test F1 = 0.76.
          </p>
          <p>
            <strong>LLM Zero-Shot:</strong> the same question asked to Gemini with no training
            examples and no retrieved context — just "is this URGENT or NORMAL?" Takes ~900 ms
            and costs ~$0.000007 per call.
          </p>
          <p>
            The comparison shows the cost of interpretability (the ML model's features are
            transparent) versus the LLM's broader language understanding.
          </p>
        </Card>

        <Card title={<>How Cost Is Calculated <Tooltip text="Cost is computed from the actual token counts returned by the API: (input_tokens × $0.075 + output_tokens × $0.30) / 1,000,000" /></>}>
          <p>
            Every Gemini API response includes the exact number of tokens used.
            We multiply by the published Gemini 2.0 Flash Lite pricing:
          </p>
          <ul className="cost-list">
            <li><span className="cost-label">Input tokens</span><span className="cost-val">$0.075 / 1M</span></li>
            <li><span className="cost-label">Output tokens</span><span className="cost-val">$0.30 / 1M</span></li>
            <li><span className="cost-label">ML Classifier</span><span className="cost-val">$0.000000 (in-process)</span></li>
          </ul>
          <p>
            At 10,000 tickets per hour the ML classifier costs $0/hr. The LLM zero-shot costs
            roughly $0.07/hr for priority classification alone. This is why production systems
            use trained classifiers for high-volume triage.
          </p>
        </Card>
      </div>

      {/* ── Honest caveats ────────────────────────────────────────────────── */}
      <section className="section">
        <h3 className="section-title">Important Caveats</h3>
        <div className="caveat-grid">
          <Caveat title="Model accuracy measures rule reproduction, not real-world urgency">
            Our labels (URGENT / NORMAL) were created by a regex keyword-scoring rule —
            not by human annotators, and not by comparing to actual airline handling records.
            The test-set accuracy of 92% means: the gradient boosting model can reproduce
            the labeling rule on unseen tweets 92% of the time. It does not mean 92% of
            predictions match what a human expert would label, or how the airline actually
            treated the case. There is no ground-truth annotation in this dataset.
          </Caveat>
          <Caveat title="Labeling is weak supervision — the model partly learns the rule">
            Because our training target (is_urgent) was created by a regex rule, the trained
            ML model will partly learn to reproduce that rule. This is expected and documented.
            The 14 engineered features (specific keyword flags, count-based stats, time mentions)
            add some independent signal that the broad labeling booleans do not capture —
            but the circularity cannot be fully eliminated without human-annotated data.
          </Caveat>
        </div>
      </section>
    </div>
  )
}

import { useState, useEffect } from 'react'
import QueryInput from './components/QueryInput'
import AnswerSection from './components/AnswerSection'
import SourcePanel from './components/SourcePanel'
import ComparisonPanel from './components/ComparisonPanel'
import OffTopicBanner from './components/OffTopicBanner'
import HowItWorksPage from './pages/HowItWorksPage'
import ModelPage from './pages/ModelPage'
import DataPage from './pages/DataPage'
import './App.css'

const API = '/api/v1'

const NAV = [
  { id: 'query',         label: 'Query' },
  { id: 'how-it-works',  label: 'How It Works' },
  { id: 'model',         label: 'Model & Metrics' },
  { id: 'data',          label: 'Data & Labels' },
]

export default function App() {
  const [page, setPage] = useState('query')
  const [query, setQuery] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [offTopicInfo, setOffTopicInfo] = useState(null)
  const [trainedMetrics, setTrainedMetrics] = useState(null)
  const [dataStats, setDataStats] = useState(null)

  useEffect(() => {
    fetch(`${API}/metrics`)
      .then(r => r.json())
      .then(d => { if (!d.status) setTrainedMetrics(d) })
      .catch(() => {})
    fetch(`${API}/data-stats`)
      .then(r => r.json())
      .then(d => { if (!d.status) setDataStats(d) })
      .catch(() => {})
  }, [])

  async function handleSubmit() {
    if (!query.trim() || loading) return
    setLoading(true)
    setError(null)
    setOffTopicInfo(null)
    setResult(null)
    try {
      const res = await fetch(`${API}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim() }),
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        // Structured off-topic block — show dedicated banner, not the generic error
        if (res.status === 422 && data.detail?.code === 'off_topic') {
          setOffTopicInfo(data.detail)
          return
        }
        // Generic error (server down, empty query, etc.)
        const msg = typeof data.detail === 'string'
          ? data.detail
          : data.detail?.message ?? `Server error ${res.status}`
        throw new Error(msg)
      }
      setResult(await res.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-inner">
          <div className="header-brand">
            <h1>Priority<span>Pulse</span></h1>
            <p>Airline Support · Decision Intelligence Assistant</p>
          </div>
          <span className="header-badge">RAG · ML · LLM</span>
        </div>
        <nav className="app-nav">
          <div className="nav-inner">
            {NAV.map(n => (
              <button
                key={n.id}
                className={`nav-tab${page === n.id ? ' active' : ''}`}
                onClick={() => setPage(n.id)}
              >
                {n.label}
              </button>
            ))}
          </div>
        </nav>
      </header>

      <main className="app-main">
        {page === 'query' && (
          <>
            <QueryInput
              value={query}
              onChange={setQuery}
              onSubmit={handleSubmit}
              loading={loading}
            />
            {loading && <div className="loading-spinner">Analysing query…</div>}
            {error && <div className="error-banner">{error}</div>}
            <OffTopicBanner info={offTopicInfo} onDismiss={() => setOffTopicInfo(null)} />
            {result && (
              <>
                <AnswerSection result={result} />
                <SourcePanel
                  sources={result.retrieved_sources}
                  lowSimilarity={result.low_similarity_warning}
                />
                <ComparisonPanel result={result} trainedMetrics={trainedMetrics} />
              </>
            )}
          </>
        )}

        {page === 'how-it-works' && <HowItWorksPage />}
        {page === 'model'        && <ModelPage trainedMetrics={trainedMetrics} />}
        {page === 'data'         && <DataPage dataStats={dataStats} />}
      </main>
    </div>
  )
}

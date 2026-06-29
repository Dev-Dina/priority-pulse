# PriorityPulse — Airline Support Decision Intelligence

A full-stack RAG application that classifies airline customer support tickets as
URGENT or NORMAL, generates LLM answers with and without retrieval context, and
displays a four-way comparison of accuracy, latency, and cost side-by-side.

Built for AIE Week 3 Bootcamp.

---

## What It Does

Every query runs four parallel processes:

| Process | What it does | Latency | Cost |
|---|---|---|---|
| **RAG Answer** | LLM with top-5 similar past tickets as context | ~1.5 s | ~$0.000072 |
| **Plain Answer** | LLM with no context (baseline) | ~4 s | ~$0.000250 |
| **ML Classifier** | Gradient Boosting on 14 text features | ~1 ms | $0.000000 |
| **LLM Zero-Shot** | Gemini asked directly: URGENT or NORMAL? | ~900 ms | ~$0.000007 |

The comparison panel shows all four results with their real latency and cost numbers,
and makes a production recommendation: use the ML classifier for high-volume triage.

---

## Architecture

```
User Query
    │
    ├─ Embed (all-MiniLM-L6-v2, 384-dim)
    │       │
    │       └─ Qdrant: cosine search over 87,848 tweet embeddings
    │               │
    ├───────────────┼─ RAG Answer  (Gemini + retrieved context)
    │               ├─ Plain Answer (Gemini, no context)
    │               ├─ ML Classifier (GradientBoosting, ~1 ms, $0)
    │               └─ LLM Zero-Shot (Gemini, binary classification)
    │
    └─ Response: answers + sources + priority labels + metrics
```

**Stack**
- Backend: FastAPI + Python 3.12
- Vector DB: Qdrant (Docker)
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- LLM: Google Gemini 2.0 Flash Lite (free tier)
- ML: scikit-learn (GradientBoosting, RandomForest, LogisticRegression)
- Frontend: React 19 + Vite (no UI library — custom CSS design system)

---

## Data

**Source:** Twitter Customer Support Corpus (TWCS) — ~1.3M tweets (Kaggle)

**After filtering:** 87,848 customer tweets to 6 US airlines:
American Airlines, Delta, Southwest, JetBlue, United, US Airways

**Filtering decisions:**
- Keep only `inbound = True` (customer→airline tweets)
- Drop airline replies — they contain the resolution, not the problem
- Drop tweets under 8 chars, bare "DM sent", "thanks", etc.
- Deduplicate on cleaned text

---

## Labeling — Weak Supervision

Labels were **not** created by human annotators. A regex keyword-scoring rule
approximates SLA-based triage:

1. **Hard override** — critical keywords (stranded, medical, sue, fraud…) → URGENT immediately
2. **Additive score** — delay signal (+3), urgency keywords (+2), profanity (+1), all-caps (+1), exclamations (+1)
3. **Threshold** — score ≥ 3 → URGENT, else NORMAL

**Result:** 18.5% URGENT (16,235), 81.5% NORMAL (71,613)

### Important caveat

The model's test-set accuracy (92%) measures **how well the model reproduces the
labeling rule on unseen tweets** — not how well it matches real-world human judgment
or actual airline handling records. There is no ground-truth annotation in this dataset.
This is documented honestly in the "Data & Labels" tab of the UI.

---

## ML Pipeline

Features extracted from raw text (14 total):
`text_length`, `word_count`, `exclamation_count`, `question_count`, `caps_ratio`,
`has_refund`, `has_cancel`, `has_delay`, `has_help`, `has_broken`,
`has_stranded`, `has_medical`, `profanity_count`, `has_time_mention`

Three models trained, best selected by **F1 score** (not accuracy — more informative
on imbalanced data):

| Model | F1 | Accuracy | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.7228 | 0.9139 | 0.8713 |
| Random Forest | 0.7205 | 0.9090 | 0.8517 |
| **Gradient Boosting** ✓ | **0.7553** | **0.9219** | **0.8803** |

_Held-out test-set metrics; best model selected by validation F1. Figures match `models/metrics.json`._

---

## Setup

### Prerequisites
- Python 3.12, uv
- Node 20+
- Docker (for Qdrant)
- A Gemini API key (free at aistudio.google.com)

### 1. Start Qdrant
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env — set GEMINI_API_KEY
```

### 3. Run the data pipeline (only once)
```bash
cd backend
uv run python scripts/run_data_pipeline.py
uv run python scripts/init_db.py
uv run python scripts/populate_vectors.py
uv run python scripts/train_models.py
```

### 4. Start the backend
```bash
cd backend
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. Start the frontend
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

---

## Docker Compose (production)

Run everything with one command (requires pre-built data pipeline outputs):

```bash
docker compose up --build
```

Opens on http://localhost:80

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/health` | GET | Qdrant status, vector count, ML model status |
| `/api/v1/metrics` | GET | Pre-computed test-set ML metrics |
| `/api/v1/query` | POST | Full pipeline — body: `{"query": "..."}` |

### Query response shape
```json
{
  "rag_answer": "...",
  "non_rag_answer": "...",
  "retrieved_sources": [{"airline": "Delta", "text": "...", "priority": "URGENT", "similarity": 0.67}],
  "low_similarity_warning": false,
  "ml_prediction": {"label": "URGENT", "confidence": 0.988, "latency_ms": 1.5, "cost_usd": 0.0},
  "llm_prediction": {"label": "URGENT", "latency_ms": 916, "cost_usd": 7e-06},
  "metrics": {"rag_ms": 1864, "rag_cost_usd": 7.2e-05, "non_rag_ms": 4340, "non_rag_cost_usd": 0.000253}
}
```

---

## Logs

All queries are appended to `logs/queries.jsonl` — one JSON object per line.
Each record includes all four outputs, latencies, costs, and a UTC timestamp.

"""
LLM Service — Gemini primary, Groq fallback.

Four responsibilities:
  1. check_topic          — topic guard: is this query within scope?
  2. answer_with_context  — RAG answer (LLM + retrieved ticket context)
  3. answer_plain         — non-RAG answer (LLM alone, baseline)
  4. classify_priority    — zero-shot label: URGENT or NORMAL

Every content call returns (text, latency_ms, cost_usd).
check_topic returns (is_allowed: bool, reason: str).

Pricing:
  Gemini 2.0 Flash Lite        — $0.075 / 1M input, $0.30 / 1M output
  Groq llama-3.3-70b-versatile — $0.59  / 1M input, $0.79 / 1M output

Groq serves only open-weight models, so the fallback's behaviour and answer
quality differ from a GPT/Claude/Gemini-proper model.
"""

import re
import time
from typing import Any, Dict, List

from google import genai
from google.genai import types
from openai import OpenAI

from app.config import settings
from app.core.logger import app_logger

# ── Pricing ($/1M tokens) ─────────────────────────────────────────────────

_GEMINI_IN  = 0.075
_GEMINI_OUT = 0.30

_GROQ_IN  = 0.59
_GROQ_OUT = 0.79


def _gemini_cost(in_tok: int, out_tok: int) -> float:
    return (in_tok * _GEMINI_IN + out_tok * _GEMINI_OUT) / 1_000_000


def _groq_cost(in_tok: int, out_tok: int) -> float:
    return (in_tok * _GROQ_IN + out_tok * _GROQ_OUT) / 1_000_000


# ── Tier-1 heuristic patterns (zero cost, <1 ms) ──────────────────────────
_HEURISTIC_BLOCKS: list[tuple[str, str]] = [
    (
        r"\b(write|generate|create|give me|show me)\s+(me\s+)?(a\s+|some\s+)?"
        r"(python|javascript|typescript|java|c\+\+|sql|html|css|bash|shell|code|"
        r"function|script|class|program|algorithm|snippet)\b",
        "code_request",
    ),
    (
        r"^\s*[\d\s\.\+\-\*\/\(\)\^%]+\s*=?\s*$",
        "arithmetic_expression",
    ),
    (
        r"\b(write|compose|give me)\s+(me\s+)?(a\s+)?(poem|song|haiku|rap|essay|story|novel|joke)\b",
        "creative_writing",
    ),
    (
        r"\b(capital\s+of|population\s+of|president\s+of|prime\s+minister\s+of)\b",
        "general_trivia",
    ),
    (
        r"\b(ignore\s+(previous|all|prior)\s+instructions|you\s+are\s+now\s+(a\s+)?DAN|"
        r"pretend\s+you\s+are|act\s+as\s+if\s+you\s+have\s+no\s+restrictions|"
        r"disregard\s+your\s+(guidelines|rules|training))\b",
        "jailbreak_attempt",
    ),
]


class LLMService:

    def __init__(self):
        self._gemini: genai.Client | None = None
        if settings.GEMINI_API_KEY:
            self._gemini = genai.Client(api_key=settings.GEMINI_API_KEY)
            app_logger.info(
                f"LLMService: Gemini active (model={settings.LLM_MODEL})"
            )
        else:
            app_logger.warning("LLMService: GEMINI_API_KEY not set — Gemini disabled.")

        self._groq: OpenAI | None = None
        if settings.GROQ_API_KEY:
            self._groq = OpenAI(
                api_key=settings.GROQ_API_KEY,
                base_url=settings.GROQ_BASE_URL,
            )
            app_logger.info(
                f"LLMService: Groq fallback active (model={settings.GROQ_MODEL})"
            )
        else:
            app_logger.warning("LLMService: GROQ_API_KEY not set — Groq fallback disabled.")

    # ── Provider calls ─────────────────────────────────────────────────────

    def _gemini_call(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, float, float]:
        """Call Gemini. Returns (text, latency_ms, cost_usd)."""
        t0 = time.perf_counter()
        response = self._gemini.models.generate_content(
            model=settings.LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        text = response.text or ""
        usage = response.usage_metadata
        cost = _gemini_cost(
            usage.prompt_token_count or 0,
            usage.candidates_token_count or 0,
        )
        return text, round(latency_ms, 2), round(cost, 6)

    def _groq_call(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, float, float]:
        """Call Groq via the OpenAI-compatible API. Returns (text, latency_ms, cost_usd)."""
        t0 = time.perf_counter()
        response = self._groq.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        text = response.choices[0].message.content or ""
        usage = response.usage
        cost = _groq_cost(usage.prompt_tokens or 0, usage.completion_tokens or 0)
        return text, round(latency_ms, 2), round(cost, 6)

    def _call(
        self,
        prompt: str,
        max_tokens: int = settings.LLM_MAX_TOKENS,
        temperature: float | None = None,
    ) -> tuple[str, float, float]:
        """Try Gemini first; fall back to Groq on any error."""
        t_val = temperature if temperature is not None else settings.LLM_TEMPERATURE

        if self._gemini is not None:
            try:
                return self._gemini_call(prompt, max_tokens, t_val)
            except Exception as e:
                app_logger.warning(
                    f"Gemini failed, falling back to Groq: {type(e).__name__}: {str(e)[:120]}"
                )

        if self._groq is not None:
            try:
                return self._groq_call(prompt, max_tokens, t_val)
            except Exception as e:
                app_logger.error(f"Groq call failed: {type(e).__name__}: {str(e)[:120]}")
                return f"[LLM unavailable: {type(e).__name__}: {str(e)[:120]}]", 0.0, 0.0

        return "[LLM unavailable: no provider configured]", 0.0, 0.0

    # ── Topic guard ────────────────────────────────────────────────────────

    def check_topic(self, query: str) -> tuple[bool, str]:
        """
        Two-tier topic guard.

        Tier 1 — heuristic regex (0 cost, <1 ms).
        Tier 2 — LLM classifier (max 5 output tokens, temperature=0).

        Fail-open: if the LLM returns anything other than an explicit "NO",
        the query is allowed through.
        """
        q = query.strip()

        for pattern, reason in _HEURISTIC_BLOCKS:
            if re.search(pattern, q, re.IGNORECASE):
                return False, reason

        guard_prompt = (
            "You are a strict topic classifier for an airline customer support chatbot.\n"
            "The chatbot ONLY handles questions about: airlines, flights, air travel, "
            "booking, reservations, cancellations, refunds, baggage, check-in, boarding, "
            "flight delays and disruptions, airport services, loyalty programmes, upgrades, "
            "and related travel support.\n\n"
            "Reply with ONLY the word YES if the message below is within that scope.\n"
            "Reply with ONLY the word NO if it is about anything else "
            "(coding, maths, politics, entertainment, general knowledge, etc.).\n\n"
            f"Message: {q[:400]}\n\n"
            "Answer (YES or NO):"
        )
        raw, _, _ = self._call(guard_prompt, max_tokens=5, temperature=0.0)

        label = raw.strip().upper()[:3]
        if label.startswith("NO"):
            return False, "off_topic"
        return True, "on_topic"

    # ── Content calls ──────────────────────────────────────────────────────

    def answer_with_context(
        self, query: str, sources: List[Dict[str, Any]]
    ) -> tuple[str, float, float]:
        """RAG answer: inject top-k retrieved tickets as context."""
        if sources:
            context_lines = []
            for s in sources:
                block = (
                    f"[{s['airline']} | {s['priority']} | similarity={s['similarity']}]\n"
                    f"Customer: {s['text']}"
                )
                if s.get("agent_response"):
                    block += f"\nResolution: {s['agent_response']}"
                context_lines.append(block)
            context = "\n\n".join(context_lines)
            prompt = (
                "You are a customer support analyst for an airline.\n"
                "Answer the user's question using the similar past tickets below as your "
                "primary source. Ground your response in how those past cases were handled. "
                "Similarity scores range from 0 to 1 — be explicit about uncertainty when "
                "scores are below 0.5. Do not invent information not present in the tickets.\n\n"
                f"--- Similar past tickets ---\n{context}\n\n"
                f"--- User question ---\n{query}\n\nAnswer:"
            )
        else:
            prompt = (
                "You are a customer support analyst for an airline. "
                "No similar past tickets were found for this query. "
                "Answer based on general airline customer support knowledge and "
                "make clear that your response is not drawn from specific past cases.\n\n"
                f"Question: {query}\n\nAnswer:"
            )

        return self._call(prompt)

    def answer_plain(self, query: str) -> tuple[str, float, float]:
        """Non-RAG answer — LLM alone, no retrieved context."""
        return self._call(query)

    def classify_priority(self, text: str) -> tuple[str, float, float]:
        """Zero-shot priority prediction: URGENT or NORMAL."""
        prompt = (
            "Classify this airline customer support ticket as exactly one of two labels:\n"
            "  URGENT — service disruption, safety concern, time-sensitive issue, "
            "or any situation needing immediate action\n"
            "  NORMAL — general inquiry, feedback, or routine request\n\n"
            "Respond with ONLY the label (URGENT or NORMAL). No explanation.\n\n"
            f"Ticket: {text}"
        )
        raw, latency_ms, cost = self._call(prompt, max_tokens=10)
        label = "URGENT" if "URGENT" in raw.upper() else "NORMAL"
        return label, latency_ms, cost


llm_service = LLMService()

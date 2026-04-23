"""
LLM Service — all Gemini API calls in one place.

Four responsibilities:
  1. check_topic          — topic guard: is this query within scope?
  2. answer_with_context  — RAG answer (LLM + retrieved ticket context)
  3. answer_plain         — non-RAG answer (LLM alone, baseline)
  4. classify_priority    — zero-shot label: URGENT or NORMAL

Every content call returns (text, latency_ms, cost_usd).
check_topic returns (is_allowed: bool, reason: str).

Pricing: Gemini 2.5 Flash Lite — $0.075 / 1M input, $0.30 / 1M output.
Free-tier usage is $0.00; these figures show relative cost at scale.
"""

import re
import time
from typing import Any, Dict, List

from google import genai
from google.genai import types

from app.config import settings


# Gemini 2.5 Flash Lite pricing ($/1M tokens)
_INPUT_PRICE = 0.075
_OUTPUT_PRICE = 0.30


def _cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * _INPUT_PRICE + output_tokens * _OUTPUT_PRICE) / 1_000_000


# ── Tier-1 heuristic patterns (zero cost, <1 ms) ──────────────────────────
# Only flag patterns that are unambiguously off-topic — err on the side of
# letting the LLM guard decide ambiguous cases rather than blocking legitimate
# support queries with overlap (e.g. "can you calculate my refund?").
_HEURISTIC_BLOCKS: list[tuple[str, str]] = [
    # explicit code-writing requests
    (
        r"\b(write|generate|create|give me|show me)\s+(me\s+)?(a\s+|some\s+)?"
        r"(python|javascript|typescript|java|c\+\+|sql|html|css|bash|shell|code|"
        r"function|script|class|program|algorithm|snippet)\b",
        "code_request",
    ),
    # homework / pure arithmetic
    (
        r"^\s*[\d\s\.\+\-\*\/\(\)\^%]+\s*=?\s*$",
        "arithmetic_expression",
    ),
    # creative writing
    (
        r"\b(write|compose|give me)\s+(me\s+)?(a\s+)?(poem|song|haiku|rap|essay|story|novel|joke)\b",
        "creative_writing",
    ),
    # explicit general-knowledge trivia unlikely to overlap with airline support
    (
        r"\b(capital\s+of|population\s+of|president\s+of|prime\s+minister\s+of)\b",
        "general_trivia",
    ),
    # jailbreak / role-play prompts
    (
        r"\b(ignore\s+(previous|all|prior)\s+instructions|you\s+are\s+now\s+(a\s+)?DAN|"
        r"pretend\s+you\s+are|act\s+as\s+if\s+you\s+have\s+no\s+restrictions|"
        r"disregard\s+your\s+(guidelines|rules|training))\b",
        "jailbreak_attempt",
    ),
]


class LLMService:

    def __init__(self):
        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def _call(
        self,
        prompt: str,
        max_tokens: int = settings.LLM_MAX_TOKENS,
        temperature: float | None = None,
    ) -> tuple[str, float, float]:
        """Raw API call. Returns (text, latency_ms, cost_usd)."""
        t_val = temperature if temperature is not None else settings.LLM_TEMPERATURE
        t0 = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model=settings.LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=t_val,
                ),
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            text = response.text
            usage = response.usage_metadata
            cost = _cost(
                usage.prompt_token_count or 0,
                usage.candidates_token_count or 0,
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            text = f"[LLM unavailable: {type(e).__name__}: {str(e)[:120]}]"
            cost = 0.0
        return text, round(latency_ms, 2), round(cost, 6)

    # ── Topic guard ────────────────────────────────────────────────────────

    def check_topic(self, query: str) -> tuple[bool, str]:
        """
        Two-tier topic guard.

        Tier 1 — heuristic regex (0 cost, <1 ms):
            Catches unambiguously off-topic patterns — coding requests, pure
            arithmetic, creative writing, jailbreak attempts.

        Tier 2 — LLM classifier (max 5 output tokens, temperature=0):
            Handles everything the heuristic passes. The prompt constrains the
            model to reply with YES (in-scope) or NO (out-of-scope).

        Fail-open policy: if the LLM call returns an error string or anything
        other than an explicit "NO", the query is allowed through. We must
        never block legitimate users because the guard malfunctions.

        Returns:
            (True,  "on_topic")      — query is within scope, proceed
            (False, reason_code)     — query is out of scope, block with reason
        """
        q = query.strip()

        # ── tier 1: heuristic ──────────────────────────────────────────────
        for pattern, reason in _HEURISTIC_BLOCKS:
            if re.search(pattern, q, re.IGNORECASE):
                return False, reason

        # ── tier 2: LLM classifier ─────────────────────────────────────────
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

        # Fail-open: block ONLY on an explicit "NO" at the start of the reply
        label = raw.strip().upper()[:3]
        if label.startswith("NO"):
            return False, "off_topic"

        return True, "on_topic"

    def answer_with_context(
        self, query: str, sources: List[Dict[str, Any]]
    ) -> tuple[str, float, float]:
        """RAG answer: inject top-k retrieved tickets as context."""
        if sources:
            context_lines = [
                f"[{s['airline']} | {s['priority']} | similarity={s['similarity']}]\n{s['text']}"
                for s in sources
            ]
            context = "\n\n".join(context_lines)
            prompt = (
                "You are a customer support analyst for an airline. "
                "Use the following similar past tickets as context to answer the user's question. "
                "If the context is not relevant, say so and answer from general knowledge.\n\n"
                f"--- Similar past tickets ---\n{context}\n\n"
                f"--- User question ---\n{query}\n\nAnswer:"
            )
        else:
            prompt = query

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

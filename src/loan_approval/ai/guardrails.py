"""Deterministic guardrails around the AI Model Analyst.

1. Scope: requests for real credit decisions or personal advice are redirected before any LLM call.
2. Privacy: questions that look like they contain personal identifiers are refused.
3. Grounding: every number with a decimal point or % sign in the answer must match a number in the
   evidence the tools returned; cited evidence IDs must be ones that were actually returned.
"""

from __future__ import annotations

import contextlib
import json
import re

SCOPE_PATTERNS = [
    r"\bshould (i|we|you|they) (approve|deny|reject|lend|fund)\b",
    r"\b(approve|deny|reject|underwrite) (this|my|our|her|his|their|the) (loan|application|mortgage|borrower|applicant)\b",
    r"\bwill (i|we|my \w+) (be|get) (approved|denied|a loan|a mortgage)\b",
    r"\b(can|could) (i|we) (get|qualify for) (a |an )?(loan|mortgage|home loan|refinance)\b",
    r"\b(my|our) (credit|loan|application|mortgage|chances|odds|dti|income)\b",
    r"\bhow (can|do|should) (i|we) (get approved|improve (my|our) (chances|odds|score))\b",
    r"\b(financial|investment|legal) advice\b",
    r"\bwhat (rate|interest rate) (will|would|should) (i|we)\b",
]
PII_PATTERNS = {
    "a social security number": r"\b\d{3}-\d{2}-\d{4}\b",
    "an account or card number": r"\b\d{9,19}\b",
    "an email address": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
    "a phone number": r"\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
}

SCOPE_REFUSAL = (
    "I can't help with real lending decisions or personal financial advice. This tool explains a "
    "research model trained on public 2025 Texas HMDA data, and its outputs are not decisions about "
    "any real person. I can explain how the model behaves, compare models, walk through the fairness "
    "audit, or explain a fictional demo scenario. A licensed lender or a HUD-approved housing "
    "counselor can help with a real application."
)


def scope_check(question: str) -> str | None:
    text = question.lower()
    for pattern in SCOPE_PATTERNS:
        if re.search(pattern, text):
            return SCOPE_REFUSAL
    return None


def pii_check(question: str) -> str | None:
    for label, pattern in PII_PATTERNS.items():
        if re.search(pattern, question):
            return (
                f"Your question looks like it contains {label}. Please don't share personal "
                "information here; this demo only works with fictional scenarios."
            )
    return None


_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?%?")


def _evidence_numbers(evidence: list[dict]) -> set[float]:
    numbers: set[float] = set()

    def walk(obj):
        if isinstance(obj, bool):
            return
        if isinstance(obj, int | float):
            numbers.add(float(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
        elif isinstance(obj, str):
            for m in _NUMBER.findall(obj):
                with contextlib.suppress(ValueError):
                    numbers.add(float(m.rstrip("%").replace(",", "")))

    for item in evidence:
        walk(item.get("data"))
    return numbers


def _matches(value: float, decimals: int, pool: set[float]) -> bool:
    """True if ``value`` equals an evidence number, either as stored or as a percentage (x100)."""
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(abs(c) - abs(value)) <= tol for n in pool for c in (n, n * 100))


def grounding_check(answer: str, evidence: list[dict], cited_ids: list[str]) -> dict:
    pool = _evidence_numbers(evidence)
    # Differences between two evidence values (e.g. a gap in percentage points) are also grounded.
    pct_pool = {round(n * 100, 6) for n in pool if abs(n) <= 1}
    diffs = {abs(a - b) for a in pct_pool for b in pct_pool} if len(pct_pool) < 400 else set()
    unverified = []
    for raw in _NUMBER.findall(answer):
        if "." not in raw and not raw.endswith("%"):
            continue  # plain integers (counts, years, list positions) are not checked
        text = raw.rstrip("%").replace(",", "")
        decimals = len(text.split(".")[1]) if "." in text else 0
        value = float(text)
        if _matches(value, decimals, pool | diffs):
            continue
        unverified.append(raw)
    returned = {e["evidence_id"] for e in evidence}
    bad_citations = [c for c in cited_ids if c not in returned]
    return {
        "grounded": not unverified and not bad_citations,
        "unverified_numbers": unverified,
        "invalid_citations": bad_citations,
    }


def wrap_untrusted(label: str, payload: dict) -> str:
    """Serialise tool output as data. Instructions inside it are not to be followed."""
    return json.dumps({"untrusted_data_label": label, "data": payload}, default=str)

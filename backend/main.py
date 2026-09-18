from __future__ import annotations

import json
import re
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from backend.llm import llm
from backend.reasoning.engine import reason


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Darukaa.Earth Causal Evidence Engine",
    version="0.5.0",
)


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class EvidenceItem(BaseModel):
    source_id: str
    metric: str | None = None
    relation: str | None = None
    effect_type: str | None = None
    effect_size: float | None = None
    effect_unit: str | None = None
    evidence_tier: str | None = None
    confidence: float | None = None
    quote: str | None = None


class EvidenceCard(BaseModel):
    intervention: str
    impacted_metrics: list[str]
    metric_coverage: float
    evidence: list[EvidenceItem]
    sources: list[dict[str, Any]]


class ChatResponse(BaseModel):
    response: str
    session_id: str
    analysis: dict[str, Any]
    evidence_cards: list[EvidenceCard]


# ============================================================
# HELPERS
# ============================================================

def extract_text_from_llm(result: Any) -> str:
    """
    Extract plain text from a LangChain LLM response.
    """

    if result is None:
        return ""

    content = getattr(result, "content", result)

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                text_value = item.get("text")

                if text_value:
                    parts.append(str(text_value))

        return "".join(parts)

    return str(content)


def clean_json_response(text: str) -> str:
    """
    Clean common markdown wrappers around JSON.
    """

    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]

    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


# ============================================================
# SITE STATE EXTRACTION
# ============================================================

def extract_site_state(message: str) -> dict:
    """
    Hybrid environmental site-state extraction.

    LLM:
        Extracts environmental information from natural language.

    Deterministic verification:
        Ensures explicit environmental keywords and numeric
        information from the user's actual message are preserved.

    The LLM does not invent environmental measurements.
    """

    prompt = f"""
You are an environmental site-state extraction system.

Extract ONLY information explicitly stated by the user.

Return JSON only.

Schema:

{{
  "soc": null,
  "moisture": null,
  "water_retention": null,
  "rainfall": null,
  "land_use": null,
  "biodiversity": null,
  "erosion": null,
  "crop": null,
  "region": null,
  "temperature": null,
  "pollution": null
}}

Rules:

- Do not invent values.
- Missing information must be null.
- SOC means soil organic carbon.
- rainfall must be numeric when explicitly given.
- Preserve qualitative descriptions such as low, poor,
  declining, dry, high, etc.
- "monoculture", "monocropping", "single crop" means
  land_use = "monoculture".
- "low biodiversity", "poor biodiversity", or
  "declining biodiversity" means biodiversity = "low".
- "poor water retention", "low water retention",
  "poor water holding capacity", or
  "low water holding capacity" means
  water_retention = "poor".
- "poor soil moisture retention" or
  "low soil moisture retention" means
  water_retention = "poor".
- "soil dries quickly" or "dries quickly" means
  water_retention = "poor".
- "high erosion", "severe erosion", or
  "increasing erosion" means erosion = "high".

Return JSON only.

User:
{message}
"""

    # --------------------------------------------------------
    # LLM EXTRACTION
    # --------------------------------------------------------

    extracted: dict[str, Any] = {}

    try:

        result = llm.invoke(prompt)

        raw = extract_text_from_llm(result)

        raw = clean_json_response(raw)

        parsed = json.loads(raw)

        if isinstance(parsed, dict):
            extracted = parsed

    except Exception:
        extracted = {}

    # --------------------------------------------------------
    # DEFAULT SCHEMA
    # --------------------------------------------------------

    expected_fields = [
        "soc",
        "moisture",
        "water_retention",
        "rainfall",
        "land_use",
        "biodiversity",
        "erosion",
        "crop",
        "region",
        "temperature",
        "pollution",
    ]

    for field in expected_fields:

        if field not in extracted:
            extracted[field] = None

    # ========================================================
    # DETERMINISTIC VERIFICATION
    # ========================================================

    text = message.lower()

    # --------------------------------------------------------
    # SOC
    # --------------------------------------------------------

    if any(
        phrase in text
        for phrase in [
            "low soil carbon",
            "low soc",
            "low soil organic carbon",
            "poor soil carbon",
            "depleted soil carbon",
            "declining soil carbon",
            "low soil organic carbon stock",
        ]
    ):
        extracted["soc"] = "low"

    # --------------------------------------------------------
    # MOISTURE
    # --------------------------------------------------------

    if any(
        phrase in text
        for phrase in [
            "low soil moisture",
            "low moisture",
            "dry soil",
            "dry field",
            "poor soil moisture",
            "declining soil moisture",
        ]
    ):
        extracted["moisture"] = "low"

    # --------------------------------------------------------
    # WATER RETENTION
    # --------------------------------------------------------

    if any(
        phrase in text
        for phrase in [
            "poor water retention",
            "low water retention",
            "poor water holding capacity",
            "low water holding capacity",
            "poor soil moisture retention",
            "low soil moisture retention",
            "soil dries quickly",
            "dries quickly",
        ]
    ):
        extracted["water_retention"] = "poor"

    # --------------------------------------------------------
    # RAINFALL
    # --------------------------------------------------------

    rainfall_match = re.search(
        r"(\d+(?:\.\d+)?)\s*mm\s*(?:of\s*)?rainfall",
        text,
    )

    if rainfall_match:

        extracted["rainfall"] = float(
            rainfall_match.group(1)
        )

    # Also support:
    #
    # rainfall is 500 mm
    # rainfall: 500 mm
    # annual rainfall 500 mm

    if extracted["rainfall"] is None:

        rainfall_match = re.search(
            r"(?:rainfall|annual rainfall)"
            r"(?:\s*(?:is|of|:))?"
            r"\s*(\d+(?:\.\d+)?)\s*mm",
            text,
        )

        if rainfall_match:

            extracted["rainfall"] = float(
                rainfall_match.group(1)
            )

    # --------------------------------------------------------
    # LAND USE / MONOCULTURE
    # --------------------------------------------------------

    if any(
        phrase in text
        for phrase in [
            "monoculture",
            "monocropping",
            "single crop",
        ]
    ):
        extracted["land_use"] = "monoculture"

    # --------------------------------------------------------
    # BIODIVERSITY
    # --------------------------------------------------------

    if any(
        phrase in text
        for phrase in [
            "low biodiversity",
            "poor biodiversity",
            "declining biodiversity",
        ]
    ):
        extracted["biodiversity"] = "low"

    # --------------------------------------------------------
    # EROSION
    # --------------------------------------------------------

    if any(
        phrase in text
        for phrase in [
            "high erosion",
            "severe erosion",
            "increasing erosion",
            "poor erosion control",
        ]
    ):
        extracted["erosion"] = "high"

    # --------------------------------------------------------
    # CLEAN UNKNOWN FIELDS
    # --------------------------------------------------------

    clean_state = {}

    for field in expected_fields:

        value = extracted.get(field)

        # Preserve only explicit values.
        if value is not None:
            clean_state[field] = value

    return clean_state


# ============================================================
# EVIDENCE CARDS
# ============================================================

def build_evidence_cards(
    reasoning_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Convert the internal reasoning result into
    frontend-ready evidence cards.
    """

    cards = []

    for recommendation in reasoning_result.get(
        "recommendations",
        [],
    ):

        evidence_items = []

        for effect in recommendation.get(
            "causal_effects",
            [],
        ):

            evidence_items.append(
                {
                    "source_id": effect.get(
                        "source_id"
                    ),

                    "metric": effect.get(
                        "metric"
                    ),

                    "relation": effect.get(
                        "relation"
                    ),

                    "effect_type": effect.get(
                        "effect_type"
                    ),

                    "effect_size": effect.get(
                        "effect_size"
                    ),

                    "effect_unit": effect.get(
                        "effect_unit"
                    ),

                    "evidence_tier": effect.get(
                        "evidence_tier"
                    ),

                    "confidence": effect.get(
                        "confidence"
                    ),

                    "quote": effect.get(
                        "quote"
                    ),
                }
            )

        sources = []

        for source in recommendation.get(
            "sources",
            [],
        ):

            sources.append(
                {
                    "id": source.get("id"),

                    "title": source.get(
                        "title"
                    ),

                    "authors": source.get(
                        "authors"
                    ),

                    "year": source.get(
                        "year"
                    ),

                    "doi": source.get(
                        "doi"
                    ),

                    "url": source.get(
                        "url"
                    ),

                    "evidence_tier": source.get(
                        "evidence_tier"
                    ),
                }
            )

        cards.append(
            {
                "intervention": recommendation.get(
                    "intervention"
                ),

                "impacted_metrics": recommendation.get(
                    "supporting_metrics",
                    [],
                ),

                # IMPORTANT:
                # This is graph coverage, not effectiveness.

                "metric_coverage": recommendation.get(
                    "metric_coverage",
                    0.0,
                ),

                "evidence": evidence_items,

                "sources": sources,
            }
        )

    return cards


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(
    message: str,
    reasoning_result: dict[str, Any],
    evidence_cards: list[dict[str, Any]],
) -> str:
    """
    Generate the user-facing scientific explanation.

    The LLM is used only for narration.

    Scientific claims must come from the supplied
    causal graph/evidence.
    """

    # Keep prompt compact to reduce LLM token usage.

    compact_cards = []

    for card in evidence_cards[:3]:

        compact_cards.append(
            {
                "intervention": card[
                    "intervention"
                ],

                "impacted_metrics": card[
                    "impacted_metrics"
                ],

                "metric_coverage": card[
                    "metric_coverage"
                ],

                "evidence": card[
                    "evidence"
                ][:5],

                "sources": [
                    {
                        "id": source.get("id"),
                        "title": source.get("title"),
                        "year": source.get("year"),
                    }
                    for source in card[
                        "sources"
                    ]
                ],
            }
        )

    prompt = f"""
You are the scientific narrator for Darukaa.Earth,
a causal evidence engine for environmental decision support.

User query:
{message}

Detected site state:
{json.dumps(
    reasoning_result.get("site_state", {}),
    default=str,
)}

Detected deficiencies:
{json.dumps(
    reasoning_result.get(
        "detected_deficiencies",
        [],
    ),
    default=str,
)}

Evidence cards:
{json.dumps(
    compact_cards,
    indent=2,
    default=str,
)}

Write a concise, scientifically grounded answer.

Rules:

1. Use ONLY information contained in the supplied
   evidence cards and site state.

2. Do not invent scientific facts.

3. Do not invent effect sizes.

4. Do not invent source attribution.

5. Preserve the direction of every reported effect.

6. Preserve context dependence when the evidence is
   context-dependent.

7. Do not combine effect sizes from different studies
   into a new number.

8. Do not treat metric_coverage as effectiveness,
   probability, confidence, or scientific superiority.

9. Mention the relevant source IDs when discussing
   quantitative or important scientific evidence.

10. If evidence is variable or conditional, explicitly
    state that.

11. If an effect is qualitative, describe it qualitatively.

12. Do not claim that an intervention is universally
    effective.

13. Do not claim that an intervention is the "best".

14. Explain:
    - what the intervention addresses
    - which environmental metrics it affects
    - what the evidence says
    - important limitations/context

15. Keep the answer understandable to a farmer or
    environmental practitioner.

Recommended structure:

**Recommendation: [intervention]**

**Addresses:**
- metric
- metric

**What the evidence says:**
Short explanation with source IDs.

**Important context:**
Short caveat if evidence varies by climate,
soil, management, crop, or other conditions.

Answer only the final user-facing response.
"""

    try:

        result = llm.invoke(prompt)

        answer = extract_text_from_llm(
            result
        ).strip()

        if answer:
            return answer

    except Exception:
        pass

    # ========================================================
    # FALLBACK
    # ========================================================

    if not evidence_cards:

        return (
            "I could not identify a sufficiently "
            "supported intervention from the current "
            "knowledge graph."
        )

    first = evidence_cards[0]

    metrics = first.get(
        "impacted_metrics",
        [],
    )

    metric_text = ", ".join(
        metrics
    )

    return (
        f"**Recommendation: "
        f"{first.get('intervention')}**\n\n"
        f"This intervention has direct evidence "
        f"connected to: {metric_text}."
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    """
    Basic API + database health check.
    """

    from sqlalchemy import text

    try:

        with __import__(
            "backend.db.database",
            fromlist=["engine"],
        ).engine.connect() as connection:

            connection.execute(
                text("SELECT 1")
            )

            nodes = connection.execute(
                text(
                    "SELECT COUNT(*) FROM node"
                )
            ).scalar()

            edges = connection.execute(
                text(
                    "SELECT COUNT(*) FROM edge"
                    " WHERE verification_status"
                    " IN ('verified','provisional')"
                )
            ).scalar()

            sources = connection.execute(
                text(
                    "SELECT COUNT(*) FROM source"
                )
            ).scalar()

            evidence_chunks = connection.execute(
                text(
                    "SELECT COUNT(*) FROM evidence_chunk"
                )
            ).scalar()

        return {
            "status": "healthy",
            "database": "connected",
            "knowledge_base": {
                "nodes": nodes,
                "edges": edges,
                "sources": sources,
                "evidence_chunks": evidence_chunks,
            },
        }

    except Exception as exc:

        return {
            "status": "unhealthy",
            "database": "error",
            "error": str(exc),
        }


# ============================================================
# CHAT
# ============================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
):

    # ========================================================
    # 1. EXTRACT SITE STATE
    # ========================================================

    site_state = extract_site_state(
        request.message
    )

    # ========================================================
    # 2. CEE REASONING
    # ========================================================

    reasoning_result = reason(
        site_state
    )

    # ========================================================
    # 3. EVIDENCE CARDS
    # ========================================================

    evidence_cards = build_evidence_cards(
        reasoning_result
    )

    # ========================================================
    # 4. GENERATE ANSWER
    # ========================================================

    answer = generate_answer(
        request.message,
        reasoning_result,
        evidence_cards,
    )

    # ========================================================
    # 5. SESSION
    # ========================================================

    session_id = (
        request.session_id
        or "session_generated"
    )

    return {
        "response": answer,

        "session_id": session_id,

        "analysis": reasoning_result,

        "evidence_cards": evidence_cards,
    }
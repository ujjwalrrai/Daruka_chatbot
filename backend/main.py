from __future__ import annotations
from fastapi.middleware.cors import CORSMiddleware
import json
import re
import uuid
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import text

from backend.db.database import engine
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
# CORS
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    why_it_works: str | None = None
    why_it_works_source: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    analysis: dict[str, Any]
    evidence_cards: list[EvidenceCard]
    needs_clarification: bool = False
    missing_fields: list[str] = []
    site_state: dict[str, Any] = {}


# ============================================================
# HELPERS
# ============================================================

def extract_text_from_llm(result: Any) -> str:
    """
    Extract plain text from a LangChain LLM response.
    """

    if result is None:
        return ""

    content = getattr(
        result,
        "content",
        result,
    )

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):

                text_value = item.get(
                    "text"
                )

                if text_value:
                    parts.append(
                        str(text_value)
                    )

        return "".join(parts)

    return str(content)


def clean_json_response(text: str) -> str:
    """Clean markdown/code fences and isolate the JSON object."""
    text = (text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    # Groq can occasionally add a short sentence before/after JSON.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]

    return text.strip()


# ============================================================
# LLM CONVERSATION / SITE-STATE PARSER
# ============================================================

ENVIRONMENT_FIELDS = [
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

CONVERSATION_INTENTS = [
    "provide_information",
    "correct_information",
    "cannot_provide_information",
    "ask_clarification",
    "follow_up_question",
    "acknowledgement",
    "unrelated",
]


def extract_site_state(
    message: str,
    previous_state: dict[str, Any] | None = None,
    missing_fields: list[str] | None = None,
) -> dict[str, Any]:
    """
    Modern conversational extraction layer.

    Groq/LangChain interprets the user's natural language and returns
    structured updates. We do NOT maintain a growing list of phrases
    such as "I don't know", "not sure", etc.

    The parser receives the previous structured state and the fields
    CEE is currently waiting for, so short replies like "500 mm",
    "mostly wheat", "I don't know", or "what do you mean by land use?"
    can be interpreted from conversational context.

    Deterministic CEE code still decides whether enough information is
    available and when to run the scientific reasoning engine.
    """

    previous_state = previous_state or {}
    missing_fields = missing_fields or []

    prompt = f"""
You are the natural-language understanding layer for Darukaa.Earth,
a causal evidence engine for environmental decision support.

Your job is ONLY to interpret the user's latest message and convert it
into a structured conversational update. Do not recommend interventions
and do not perform scientific reasoning.

PREVIOUS SITE STATE:
{json.dumps(previous_state, indent=2, default=str)}

FIELDS CEE IS CURRENTLY WAITING FOR:
{json.dumps(missing_fields)}

LATEST USER MESSAGE:
{message}

Return JSON ONLY in exactly this shape:
{{
  "intent": "provide_information",
  "state_updates": {{
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
  }},
  "cannot_provide": [],
  "user_question": null
}}

INTENT VALUES:
- provide_information: the user supplied environmental information.
- correct_information: the user is correcting a previous value.
- cannot_provide_information: the user explicitly says they do not know,
  cannot provide, are unsure about, or otherwise cannot answer a requested
  environmental field.
- ask_clarification: the user is asking what a requested field means or
  asking the assistant to clarify the question.
- follow_up_question: the user asks a question about the recommendation,
  evidence, metric, intervention, or scientific reasoning.
- acknowledgement: short conversational acknowledgement with no new fact.
- unrelated: the message is unrelated to the environmental conversation.

STATE UPDATE RULES:
1. Extract ONLY facts explicitly stated or clearly implied by the latest
   user message in the context of the requested field.
2. Never invent measurements, crop types, land use, locations, or other
   environmental facts.
3. A short answer such as "500 mm" should populate rainfall when rainfall
   is one of the fields CEE is waiting for.
4. A short answer such as "monoculture wheat" should populate land_use
   and crop when appropriate.
5. If the user says they don't know / are unsure / cannot provide a field,
   do NOT keep asking them for the same field. Put that field name in
   cannot_provide and leave its state_update null.
6. If the user asks "what do you mean by land use?", use intent
   ask_clarification and do not mark land_use as unknown.
7. If the user corrects an earlier fact, return the corrected value.
8. Do not copy unchanged values from PREVIOUS SITE STATE into
   state_updates. Return only NEW or CORRECTED values from this turn.
9. rainfall should be numeric in millimetres when the user provides a
   numerical rainfall value.
10. For qualitative environmental states, preserve the user's meaning
    with concise values such as low, high, poor, declining, etc.
11. If the user provides a crop but no land-use statement, set crop only.
12. Use null for fields that were not updated.
13. cannot_provide must contain canonical field names from the schema,
    for example ["land_use"] or ["rainfall"].
14. user_question should contain a concise description of the user's
    question only when intent is ask_clarification or follow_up_question;
    otherwise use null.

Return JSON only. No markdown. No explanation outside the JSON.
"""

    default_updates = {field: None for field in ENVIRONMENT_FIELDS}
    fallback = {
        "intent": "acknowledgement",
        "state_updates": default_updates,
        "cannot_provide": [],
        "user_question": None,
    }

    try:
        result = llm.invoke(prompt)
        raw = clean_json_response(extract_text_from_llm(result))
        parsed = json.loads(raw)

        if not isinstance(parsed, dict):
            return fallback

        intent = str(parsed.get("intent") or "acknowledgement").strip()
        if intent not in CONVERSATION_INTENTS:
            intent = "acknowledgement"

        raw_updates = parsed.get("state_updates")
        if not isinstance(raw_updates, dict):
            raw_updates = {}

        updates = {field: None for field in ENVIRONMENT_FIELDS}
        for field in ENVIRONMENT_FIELDS:
            value = raw_updates.get(field)
            if value is not None:
                updates[field] = value

        cannot_provide = parsed.get("cannot_provide")
        if not isinstance(cannot_provide, list):
            cannot_provide = []
        cannot_provide = [
            str(field).strip()
            for field in cannot_provide
            if str(field).strip() in ENVIRONMENT_FIELDS
        ]

        user_question = parsed.get("user_question")
        if user_question is not None:
            user_question = str(user_question).strip() or None

        # Safety invariant: if the parser says the user cannot provide a
        # field, that field must not simultaneously receive a value.
        for field in cannot_provide:
            updates[field] = None

        return {
            "intent": intent,
            "state_updates": updates,
            "cannot_provide": cannot_provide,
            "user_question": user_question,
        }

    except Exception:
        return fallback


def normalize_unknown_fields(
    previous_state: dict[str, Any],
    cannot_provide: list[str],
) -> dict[str, Any]:
    """
    Convert an explicit inability to answer into persistent uncertainty.

    "unknown" means "the user explicitly cannot provide this information".
    It is intentionally different from None, which means "we have not asked
    for / received this information yet".
    """
    updates: dict[str, Any] = {}
    for field in cannot_provide:
        if field in ENVIRONMENT_FIELDS and previous_state.get(field) is None:
            updates[field] = "unknown"
    return updates


def merge_site_state(
    previous: dict[str, Any],
    new_values: dict[str, Any],
) -> dict[str, Any]:
    """Merge explicit new facts over persisted facts; never erase old facts."""
    merged = dict(previous or {})

    for key, value in (new_values or {}).items():
        if value is not None:
            merged[key] = value

    return merged


# ============================================================
# CONVERSATIONAL SESSION STATE
# ============================================================

# Critical context needed before CEE gives a soil-management
# recommendation. These are deliberately small: ask only for
# information that materially changes graph interpretation.
CRITICAL_CONTEXT = {
    "soc": ["rainfall", "land_use"],
    "water_retention": ["rainfall", "land_use"],
    "biodiversity": ["land_use"],
    "erosion": ["rainfall", "land_use"],
}

FIELD_LABELS = {
    "soc": "soil organic carbon (SOC)",
    "rainfall": "approximate annual rainfall (mm)",
    "land_use": "current land use or cropping system",
    "water_retention": "water retention / water holding capacity",
    "biodiversity": "biodiversity status",
    "erosion": "erosion status",
    "crop": "main crop",
    "region": "region",
}


def ensure_session_table() -> None:
    """Add persistent structured state to the existing session table."""
    with engine.begin() as connection:
        connection.execute(
            text(
                "ALTER TABLE session "
                "ADD COLUMN IF NOT EXISTS site_state JSONB NOT NULL DEFAULT '{}'::jsonb"
            )
        )


def get_or_create_session(session_id: str | None) -> str:
    """Return a stable UUID session and create it when necessary."""
    ensure_session_table()

    if session_id:
        try:
            sid = str(uuid.UUID(str(session_id)))
        except (ValueError, AttributeError):
            sid = str(uuid.uuid4())
    else:
        sid = str(uuid.uuid4())

    with engine.begin() as connection:
        row = connection.execute(
            text(
                "SELECT id FROM session "
                "WHERE id = CAST(:id AS uuid)"
            ),
            {"id": sid},
        ).first()

        if row is None:
            connection.execute(
                text(
                    "INSERT INTO session "
                    "(id, label, site_state) "
                    "VALUES "
                    "(CAST(:id AS uuid), :label, CAST(:state AS jsonb))"
                ),
                {
                    "id": sid,
                    "label": "Darukaa.Earth session",
                    "state": "{}",
                },
            )

    return sid


def load_session_state(session_id: str) -> dict[str, Any]:
    ensure_session_table()
    with engine.connect() as connection:
        row = connection.execute(
            text("SELECT site_state FROM session WHERE id = CAST(:id AS uuid)"),
            {"id": session_id},
        ).scalar_one_or_none()
    return dict(row or {})



def save_session_state(session_id: str, site_state: dict[str, Any]) -> None:
    ensure_session_table()
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE session "
                "SET site_state = CAST(:state AS jsonb), updated_at = NOW() "
                "WHERE id = CAST(:id AS uuid)"
            ),
            {"id": session_id, "state": json.dumps(site_state, default=str)},
        )


def detect_missing_context(site_state: dict[str, Any]) -> list[str]:
    """Determine whether the field context is sufficient for CEE reasoning."""
    problems = []
    if site_state.get("soc") is not None:
        problems.append("soc")
    if site_state.get("water_retention") is not None:
        problems.append("water_retention")
    if site_state.get("biodiversity") is not None:
        problems.append("biodiversity")
    if site_state.get("erosion") is not None:
        problems.append("erosion")

    if not problems:
        # No environmental problem has been identified yet.
        return ["environmental_problem"]

    required: list[str] = []
    for problem in problems:
        for field in CRITICAL_CONTEXT.get(problem, []):
            value = site_state.get(field)
            # None = not yet known and should be requested.
            # "unknown" = user explicitly cannot provide it; proceed with
            # uncertainty instead of asking the same question forever.
            if value is None and field not in required:
                required.append(field)
    return required


def clarification_question(
    missing_fields: list[str],
    site_state: dict[str, Any],
    follow_up: bool = False,
) -> str:
    labels = [
        FIELD_LABELS.get(field, field.replace("_", " "))
        for field in missing_fields
    ]

    if not labels:
        return ""

    # After a user has answered a clarification, ask only for the
    # next missing item. This prevents the assistant from repeating
    # the same two-part question after every short reply.
    if follow_up:
        return (
            f"Thanks — I have that. "
            f"What is your {labels[0]}?"
        )

    if len(labels) == 1:
        return f"To make this recommendation context-specific, what is your {labels[0]}?"

    if len(labels) == 2:
        return (
            f"To make this recommendation context-specific, "
            f"what are your {labels[0]} and {labels[1]}?"
        )

    return (
        "To make this recommendation context-specific, "
        "could you provide: " + ", ".join(labels) + "?"
    )



def route_conversational_turn(
    message: str,
    previous_state: dict[str, Any],
    missing_fields: list[str],
) -> dict[str, Any]:
    """Use Groq as the conversational router when CEE is waiting for context.

    This is deliberately semantic rather than phrase matching. It decides
    whether the latest turn is an answer, an explicit inability to answer,
    a request for clarification, a follow-up question, or unrelated chat.
    """
    pending = [FIELD_LABELS.get(f, f.replace("_", " ")) for f in missing_fields]
    prompt = f"""
You are the conversational router for an environmental scientist assistant.
Do NOT do scientific reasoning. Do NOT invent environmental facts.

Previous site state:
{json.dumps(previous_state, default=str)}

Information currently needed before the scientific engine can run:
{json.dumps(missing_fields)}

The assistant's pending question is approximately:
{", ".join(pending) if pending else "none"}

Latest user message:
{message}

Classify the user's latest turn semantically. Return JSON ONLY:
{{
  "route": "answer_context",
  "field_updates": {{}},
  "cannot_provide": [],
  "reply": null
}}

Allowed route values:
- answer_context: the user supplied information that can answer the pending question.
- cannot_answer: the user explicitly cannot provide the requested information.
- clarify: the user does not understand the assistant's question and wants it explained.
- follow_up: the user asks a question or asks for a recommendation/explanation instead of answering the pending question.
- unrelated: the message is unrelated to the environmental conversation.
- acknowledgement: simple acknowledgement with no substantive request.

Rules:
1. Understand meaning, not exact phrases. Examples are illustrative, not a list to match.
2. If the user says they do not know, are unsure, cannot answer, or declines to provide a requested field, use cannot_answer and put the relevant canonical field(s) in cannot_provide.
3. If the user asks "what?", "why?", or otherwise signals confusion about the pending question, use clarify and draft a natural short reply explaining the pending question.
4. If the user asks for a recommendation while context is incomplete, use follow_up and draft a natural reply that explains what is still needed and that the user may say they do not know.
5. If the message contains a real environmental value, use answer_context and extract only explicitly provided values into field_updates.
6. Never fabricate a value just to satisfy the pending question.
7. reply must be a user-facing response only for clarify, follow_up, unrelated, or acknowledgement. For answer_context/cannot_answer it may be null.
8. Use canonical field names such as rainfall, land_use, crop, soc.
"""
    fallback = {"route": "answer_context", "field_updates": {}, "cannot_provide": [], "reply": None}
    try:
        result = llm.invoke(prompt)
        raw = clean_json_response(extract_text_from_llm(result))
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return fallback
        route = str(parsed.get("route") or "answer_context").strip()
        allowed = {"answer_context", "cannot_answer", "clarify", "follow_up", "unrelated", "acknowledgement"}
        if route not in allowed:
            route = "answer_context"
        updates = parsed.get("field_updates") if isinstance(parsed.get("field_updates"), dict) else {}
        cannot = parsed.get("cannot_provide") if isinstance(parsed.get("cannot_provide"), list) else []
        cannot = [str(x).strip() for x in cannot if str(x).strip() in ENVIRONMENT_FIELDS]
        reply = parsed.get("reply")
        return {
            "route": route,
            "field_updates": {k: v for k, v in updates.items() if k in ENVIRONMENT_FIELDS and v is not None},
            "cannot_provide": cannot,
            "reply": str(reply).strip() if reply else None,
        }
    except Exception:
        return fallback


# ============================================================
# EVIDENCE CARDS
# ============================================================

def _mechanism_explanation(recommendation: dict[str, Any]) -> str | None:
    """Return a mechanism explanation only when the CEE graph has one."""
    paths = recommendation.get("mechanism_paths") or []
    if not paths:
        return None

    explanations = []
    seen = set()
    for path in paths[:3]:
        mechanisms = path.get("mechanisms") or []
        final_metric = path.get("final_metric")
        if not mechanisms:
            continue
        mechanism_text = " → ".join(mechanisms)
        if final_metric:
            text_value = f"The causal graph links the intervention through {mechanism_text} toward {final_metric}."
        else:
            text_value = f"The causal graph links the intervention through {mechanism_text}."
        if text_value not in seen:
            explanations.append(text_value)
            seen.add(text_value)
    return " ".join(explanations) if explanations else None


def generate_llm_mechanism_explanations(
    evidence_cards: list[dict[str, Any]],
    site_state: dict[str, Any],
) -> dict[str, str]:
    """
    Generate general scientific mechanism explanations only for cards
    where the CEE graph does not currently contain a mechanism path.

    These explanations are explicitly model-generated and must never
    masquerade as retrieved evidence or citations.
    """
    cards_needing_explanation = [
        card for card in evidence_cards
        if not card.get("why_it_works")
    ]
    if not cards_needing_explanation:
        return {}

    payload = []
    for card in cards_needing_explanation[:3]:
        payload.append({
            "intervention": card.get("intervention"),
            "impacted_metrics": card.get("impacted_metrics", []),
            "evidence": [
                {
                    "metric": item.get("metric"),
                    "relation": item.get("relation"),
                    "quote": item.get("quote"),
                }
                for item in card.get("evidence", [])[:5]
            ],
        })

    prompt = f"""
You are a scientific explainer for Darukaa.Earth.

CEE has already performed the causal reasoning and evidence retrieval.
Some recommendations do not yet have an explicit mechanism path in the
CEE graph. For those recommendations, provide a short GENERAL SCIENTIFIC
EXPLANATION of why the intervention could affect the listed metrics.

Site context:
{json.dumps(site_state, default=str)}

Recommendations needing explanation:
{json.dumps(payload, indent=2, default=str)}

Return JSON only in this exact shape:
{{
  "intervention_name": "2-4 sentence scientific explanation"
}}

STRICT RULES:
1. This is general scientific knowledge, NOT retrieved evidence.
2. Do NOT invent, quote, or cite papers.
3. Do NOT attribute any statement to an evidence source or source ID.
4. Do NOT invent numerical effect sizes, percentages, units, dates, or study results.
5. Do NOT contradict the supplied evidence.
6. Respect context: use words such as "can", "may", or "is expected to"
   when the mechanism depends on conditions.
7. Explain the biological/ecological/soil process, not whether the
   intervention is "best" or universally effective.
8. Keep each explanation concise and understandable to a practitioner.
9. If a scientifically reasonable mechanism cannot be explained safely,
   return an empty string for that intervention.
"""

    try:
        result = llm.invoke(prompt)
        raw = clean_json_response(extract_text_from_llm(result))
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {
                str(k): str(v).strip()
                for k, v in parsed.items()
                if v and str(v).strip()
            }
    except Exception:
        pass
    return {}


def build_evidence_cards(
    reasoning_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert internal CEE reasoning into frontend-ready evidence cards."""
    cards = []

    for recommendation in reasoning_result.get("recommendations", []):
        evidence_items = []
        for effect in recommendation.get("causal_effects", []):
            evidence_items.append({
                "source_id": effect.get("source_id"),
                "metric": effect.get("metric"),
                "relation": effect.get("relation"),
                "effect_type": effect.get("effect_type"),
                "effect_size": effect.get("effect_size"),
                "effect_unit": effect.get("effect_unit"),
                "evidence_tier": effect.get("evidence_tier"),
                "confidence": effect.get("confidence"),
                "quote": effect.get("quote"),
            })

        sources = []
        for source in recommendation.get("sources", []):
            sources.append({
                "id": source.get("id"),
                "title": source.get("title"),
                "authors": source.get("authors"),
                "year": source.get("year"),
                "doi": source.get("doi"),
                "url": source.get("url"),
                "evidence_tier": source.get("evidence_tier"),
            })

        why_it_works = _mechanism_explanation(recommendation)

        cards.append({
            "intervention": recommendation.get("intervention"),
            "impacted_metrics": recommendation.get("supporting_metrics", []),
            "metric_coverage": recommendation.get("metric_coverage", 0.0),
            "evidence": evidence_items,
            "sources": sources,
            "why_it_works": why_it_works,
            "why_it_works_source": "cee_mechanism" if why_it_works else None,
        })

    return cards


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(
    message: str,
    reasoning_result: dict[str, Any],
    evidence_cards: list[dict[str, Any]],
    previous_state: dict[str, Any] | None = None,
) -> str:
    """
    Generate the user-facing scientific explanation.

    The LLM is used only for narration.

    Scientific claims must come from the supplied
    causal graph/evidence.
    """

    # First fill the explanation gap for recommendations whose
    # CEE graph has no explicit mechanism path.
    llm_mechanisms = generate_llm_mechanism_explanations(
        evidence_cards,
        reasoning_result.get("site_state", {}),
    )

    for card in evidence_cards:
        if not card.get("why_it_works"):
            generated = llm_mechanisms.get(str(card.get("intervention")), "")
            if generated:
                card["why_it_works"] = generated
                card["why_it_works_source"] = "llm_scientific_explanation"

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

                "why_it_works": card.get("why_it_works"),
                "why_it_works_source": card.get("why_it_works_source"),

                "evidence": card[
                    "evidence"
                ][:5],

                "sources": [
                    {
                        "id": source.get(
                            "id"
                        ),

                        "title": source.get(
                            "title"
                        ),

                        "year": source.get(
                            "year"
                        ),
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
    reasoning_result.get(
        "site_state",
        {},
    ),
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

Respond like a knowledgeable environmental scientist in an ongoing
conversation, NOT like a form and NOT like a generic chatbot.

The user's latest message is the thing you are answering.

Rules:

1. Treat retrieved CEE evidence and CEE graph mechanisms as the
   authoritative evidence layer.

2. A card with why_it_works_source = "cee_mechanism" is grounded in
   the CEE causal graph.

3. A card with why_it_works_source = "llm_scientific_explanation"
   is GENERAL SCIENTIFIC KNOWLEDGE generated by the LLM because the
   current CEE graph did not contain an explicit mechanism. Clearly
   describe it as an LLM-generated scientific explanation when useful.

4. Never present an LLM-generated explanation as retrieved evidence.

5. Do not invent evidence, citations, source attribution, effect sizes,
   study results, percentages, or units.

3. Preserve context dependence and conditional effects.

4. Do not combine effect sizes from different studies into a new number.

5. Do not treat metric_coverage as effectiveness, probability,
   confidence, or scientific superiority.

6. If the user asks a follow-up about a specific intervention,
   metric, effect, source, limitation, or comparison, answer THAT
   question directly using the supplied evidence instead of blindly
   repeating the recommendation template.

7. If the user says something conversational such as "okay",
   "why?", "tell me more", or "what does that mean?", respond
   naturally using the current analysis context.

8. If the user asks why an intervention was recommended, explain the
   chain from their site condition -> detected deficiency ->
   intervention -> affected metrics -> evidence.

9. If the evidence is variable or conditional, say so explicitly.

10. If an effect is qualitative, describe it qualitatively.

11. Never claim an intervention is universally effective or "the best".

12. Mention source IDs for quantitative or important scientific claims.

13. Keep answers concise but useful. Avoid repeating information the
    user already knows.

Conversation context:
- Previous site state:
{json.dumps(previous_state or {}, default=str)}
- Current site state:
{json.dumps(reasoning_result.get("site_state", {}), default=str)}

Recommended answer style:
Use normal conversational paragraphs and short bullets where useful.
Only use a "Recommendation / Addresses / Evidence" structure when it
actually helps answer the user's latest message.

Answer only the final user-facing response.
"""

    try:

        result = llm.invoke(
            prompt
        )

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
# ANALYZE
# ============================================================

@app.post("/analyze")
def analyze(
    site_state: dict[str, Any],
):
    """
    Direct CEE reasoning endpoint.

    Accepts structured environmental site-state data
    and runs the causal reasoning engine.

    Pipeline:

        Site State
             ↓
        Deficiency Detection
             ↓
        Causal Graph
             ↓
        Intervention Candidates
             ↓
        Hybrid Evidence Retrieval
             ↓
        Evidence-Backed Analysis
    """

    reasoning_result = reason(
        site_state
    )

    # ========================================================
    # SCIENTIST NARRATION
    # ========================================================
    # The causal reasoning engine remains the scientific
    # authority. Groq is used only to turn the grounded
    # reasoning and evidence into a readable response.
    evidence_cards = build_evidence_cards(
        reasoning_result
    )

    scientist_response = generate_answer(
        json.dumps(site_state, default=str),
        reasoning_result,
        evidence_cards,
    )

    reasoning_result["scientist_response"] = scientist_response
    reasoning_result["evidence_cards"] = evidence_cards

    return reasoning_result


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
    """Multi-turn environmental scientist conversation.

    Persistent state lives in PostgreSQL. LangChain/Groq is used
    for extraction and narration; the CEE causal graph remains the
    scientific reasoning authority.
    """

    # 1. Resolve a persistent session.
    session_id = get_or_create_session(request.session_id)

    # 2. Load state accumulated from earlier turns.
    previous_state = load_session_state(session_id)

    # 3. Let the LLM interpret the latest conversational turn.
    #    It receives the previous state and the fields CEE is waiting for,
    #    so we do not need a growing collection of phrase-specific rules.
    previous_missing = detect_missing_context(previous_state)
    parsed_turn = extract_site_state(
        request.message,
        previous_state=previous_state,
        missing_fields=previous_missing,
    )

    turn_state = parsed_turn.get("state_updates", {})

    # An explicit "I don't know" becomes persistent uncertainty rather than
    # an endlessly missing field.
    unknown_updates = normalize_unknown_fields(
        previous_state,
        parsed_turn.get("cannot_provide", []),
    )
    turn_state.update(unknown_updates)

    # 4. Merge new facts with the previous field state.
    site_state = merge_site_state(previous_state, turn_state)
    save_session_state(session_id, site_state)

    # 5. When CEE is waiting for context, give Groq a conversational routing
    # pass. This prevents messages such as "what?", "why?", "I don't know",
    # or "recommend me" from being treated as if the user simply failed to
    # answer the same question.
    pending_for_router = detect_missing_context(site_state)
    router = route_conversational_turn(
        request.message,
        site_state,
        pending_for_router,
    ) if pending_for_router else {"route": "answer_context", "field_updates": {}, "cannot_provide": [], "reply": None}

    if router.get("field_updates"):
        site_state = merge_site_state(site_state, router["field_updates"])
        save_session_state(session_id, site_state)
        missing_fields = detect_missing_context(site_state)
    else:
        missing_fields = pending_for_router

    if router.get("cannot_provide"):
        unknown_updates = normalize_unknown_fields(site_state, router["cannot_provide"])
        site_state = merge_site_state(site_state, unknown_updates)
        save_session_state(session_id, site_state)
        missing_fields = detect_missing_context(site_state)

    # Conversational turns are answered by Groq instead of being forced into
    # the scientific missing-context question.
    if router.get("route") in {"clarify", "follow_up", "unrelated", "acknowledgement"}:
        answer = router.get("reply") or "Sure — tell me what you would like to know."
        return {
            "response": answer,
            "session_id": session_id,
            "analysis": {
                "site_state": site_state,
                "detected_deficiencies": [],
                "recommendations": [],
                "clarification": {
                    "needed": bool(missing_fields),
                    "missing_fields": missing_fields,
                    "question": answer if router.get("route") == "clarify" else None,
                },
                "conversation_intent": router.get("route"),
            },
            "evidence_cards": [],
            "needs_clarification": bool(missing_fields),
            "missing_fields": missing_fields,
            "site_state": site_state,
        }

    # 6. Handle the original parser's clarification classification as a
    # fallback for models that classify the same turn differently.
    if parsed_turn.get("intent") == "ask_clarification":
        field = (previous_missing or ["land_use"])[0]
        if field == "land_use":
            answer = (
                "By land use, I mean how the field is currently managed — "
                "for example monocropping, mixed/intercropping, pasture, "
                "agroforestry, or fallow. If you do not know, just say so "
                "and I can continue with that context marked as unknown."
            )
        else:
            answer = (
                f"By {FIELD_LABELS.get(field, field.replace('_', ' '))}, "
                "I mean the current condition or value of that part of the site. "
                "If you do not know it, you can say so and I will continue with "
                "that uncertainty explicitly recorded."
            )

        return {
            "response": answer,
            "session_id": session_id,
            "analysis": {
                "site_state": site_state,
                "detected_deficiencies": [],
                "recommendations": [],
                "clarification": {
                    "needed": True,
                    "missing_fields": detect_missing_context(site_state),
                    "question": answer,
                },
                "conversation_intent": parsed_turn.get("intent"),
            },
            "evidence_cards": [],
            "needs_clarification": bool(detect_missing_context(site_state)),
            "missing_fields": detect_missing_context(site_state),
            "site_state": site_state,
        }

    # 6. Ask for genuinely missing context instead of producing a premature
    #    scientific recommendation.
    missing_fields = detect_missing_context(site_state)

    if missing_fields:
        question = clarification_question(
            missing_fields,
            site_state,
            follow_up=bool(previous_state),
        )
        analysis = {
            "site_state": site_state,
            "detected_deficiencies": [],
            "recommendations": [],
            "clarification": {
                "needed": True,
                "missing_fields": missing_fields,
                "question": question,
            },
        }
        return {
            "response": question,
            "session_id": session_id,
            "analysis": analysis,
            "evidence_cards": [],
            "needs_clarification": True,
            "missing_fields": missing_fields,
            "site_state": site_state,
        }

    # 7. Enough context exists: run the real CEE reasoning engine.
    reasoning_result = reason(site_state)

    # 8. Convert grounded graph evidence into frontend cards.
    evidence_cards = build_evidence_cards(reasoning_result)

    # 9. Groq/LangChain narrates only the supplied scientific evidence.
    answer = generate_answer(
        request.message,
        reasoning_result,
        evidence_cards,
        previous_state=previous_state,
    )

    reasoning_result["scientist_response"] = answer
    reasoning_result["conversation"] = {
        "session_id": session_id,
        "turn_state": turn_state,
        "merged_site_state": site_state,
        "reused_previous_context": bool(previous_state),
        "intent": parsed_turn.get("intent"),
        "cannot_provide": parsed_turn.get("cannot_provide", []),
    }

    return {
        "response": answer,
        "session_id": session_id,
        "analysis": reasoning_result,
        "evidence_cards": evidence_cards,
        "needs_clarification": False,
        "missing_fields": [],
        "site_state": site_state,
    }

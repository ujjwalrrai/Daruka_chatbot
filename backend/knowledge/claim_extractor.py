import json
import re
import time

from backend.llm import llm
from backend.knowledge.claim_schema import (
    ClaimExtractionResult,
    ExtractedClaim,
)


# ============================================================
# CONFIG
# ============================================================

MAX_RETRIES = 2

# Gemini retry delays
RETRY_DELAYS = (5, 10)


# ============================================================
# HELPERS
# ============================================================

def _extract_json(raw_text: str) -> dict:
    """
    Extract the first JSON object from the model response.

    Handles:
    - plain JSON
    - markdown code fences
    - surrounding text
    """

    if not raw_text:
        raise ValueError("Model returned empty response.")

    text = raw_text.strip()

    # --------------------------------------------------------
    # Remove markdown code fences
    # --------------------------------------------------------

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    text = text.strip()

    # --------------------------------------------------------
    # Direct JSON
    # --------------------------------------------------------

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # Find JSON object inside surrounding text
    # --------------------------------------------------------

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "Could not find a JSON object in model response."
        )

    json_text = text[start:end + 1]

    try:
        return json.loads(json_text)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON returned by model: {exc}"
        ) from exc


# ============================================================
# NODE HELPERS
# ============================================================

def _node_ids(available_nodes: list[dict]) -> set[str]:
    """
    Build the set of canonical node IDs allowed in the graph.
    """

    return {
        str(node["id"])
        for node in available_nodes
        if node.get("id")
    }


# ============================================================
# CLAIM VALIDATION
# ============================================================

def _validate_claim(
    claim: ExtractedClaim,
    valid_nodes: set[str],
) -> ExtractedClaim | None:
    """
    Validate one extracted scientific claim against the ontology.
    """

    # --------------------------------------------------------
    # Node validation
    # --------------------------------------------------------

    if claim.src_node not in valid_nodes:
        print(
            f"SKIP claim: unknown src_node={claim.src_node}"
        )
        return None

    if claim.dst_node not in valid_nodes:
        print(
            f"SKIP claim: unknown dst_node={claim.dst_node}"
        )
        return None

    # --------------------------------------------------------
    # Self-loop protection
    # --------------------------------------------------------

    if claim.src_node == claim.dst_node:
        print(
            f"SKIP self-loop: {claim.src_node}"
        )
        return None

    # --------------------------------------------------------
    # Relation validation
    # --------------------------------------------------------

    valid_relations = {
        "increases",
        "decreases",
        "enables",
        "requires",
        "antagonises",
        "no_effect",
    }

    if claim.relation not in valid_relations:
        print(
            f"SKIP invalid relation={claim.relation}"
        )
        return None

    # --------------------------------------------------------
    # Effect type validation
    # --------------------------------------------------------

    valid_effect_types = {
        "pct_change",
        "absolute",
        "log_response_ratio",
        "qualitative",
    }

    if claim.effect_type not in valid_effect_types:
        print(
            f"SKIP invalid effect_type={claim.effect_type}"
        )
        return None

    # --------------------------------------------------------
    # Qualitative claims must not carry numeric effect size
    # --------------------------------------------------------

    if claim.effect_type == "qualitative":
        claim.effect_size = None
        claim.effect_unit = None

    # --------------------------------------------------------
    # no_effect should not have a numeric effect size
    # --------------------------------------------------------

    if claim.relation == "no_effect":
        claim.effect_size = None
        claim.effect_unit = None

    # --------------------------------------------------------
    # Quote is mandatory for provenance
    # --------------------------------------------------------

    if not claim.quote or not claim.quote.strip():
        print("SKIP claim: empty quote")
        return None

    # --------------------------------------------------------
    # Confidence sanity
    # --------------------------------------------------------

    if not 0.0 <= claim.confidence <= 1.0:
        print(
            f"SKIP claim: invalid confidence={claim.confidence}"
        )
        return None

    # --------------------------------------------------------
    # Effect direction sanity
    # --------------------------------------------------------

    if (
        claim.effect_type in {
            "pct_change",
            "absolute",
            "log_response_ratio",
        }
        and claim.effect_size is not None
    ):

        if claim.relation == "increases":
            if claim.effect_size < 0:
                print(
                    "SKIP claim: increases relationship has "
                    f"negative effect_size={claim.effect_size}"
                )
                return None

        elif claim.relation == "decreases":
            if claim.effect_size > 0:
                print(
                    "SKIP claim: decreases relationship has "
                    f"positive effect_size={claim.effect_size}"
                )
                return None

    return claim


# ============================================================
# GEMINI CONTENT NORMALIZATION
# ============================================================

def _normalize_response_content(raw_content) -> str:
    """
    Normalize LangChain Gemini response content.

    Gemini may return:

        "text"

    or:

        [
            {
                "type": "text",
                "text": "..."
            }
        ]

    """

    if raw_content is None:
        return ""

    # Simple string
    if isinstance(raw_content, str):
        return raw_content

    # Gemini/LangChain content blocks
    if isinstance(raw_content, list):

        parts = []

        for item in raw_content:

            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):

                if item.get("type") == "text":
                    text_value = item.get("text")

                    if text_value:
                        parts.append(str(text_value))

                elif "text" in item:
                    parts.append(
                        str(item["text"])
                    )

        return "\n".join(parts)

    return str(raw_content)


# ============================================================
# MAIN EXTRACTION FUNCTION
# ============================================================

def extract_claims(
    evidence_text: str,
    source_id: str,
    available_nodes: list[dict],
) -> list[ExtractedClaim]:
    """
    Extract evidence-backed causal claims from one evidence chunk.

    Parameters
    ----------
    evidence_text:
        Scientific evidence passage.

    source_id:
        ID of the source containing the passage.

    available_nodes:
        Canonical ontology nodes currently available in the DB.

    Returns
    -------
    list[ExtractedClaim]

    Returns [] when the passage contains no extractable
    causal claim.

    Raises
    ------
    RuntimeError
        When the LLM fails after all retry attempts.
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not evidence_text or not evidence_text.strip():
        return []

    valid_nodes = _node_ids(available_nodes)

    # --------------------------------------------------------
    # Build compact ontology representation
    #
    # We intentionally do NOT send full descriptions.
    # This reduces prompt size and keeps Gemini focused.
    # --------------------------------------------------------

    node_lines = []

    for node in available_nodes:

        node_id = node.get("id")
        kind = node.get("kind")
        name = node.get("canonical_name")

        if not node_id:
            continue

        node_lines.append(
            f"- {node_id} | {kind} | {name}"
        )

    node_text = "\n".join(node_lines)

    # --------------------------------------------------------
    # System prompt
    # --------------------------------------------------------

    system_prompt = """
You are the scientific evidence extraction component of
Darukaa.Earth CEE (Causal Evidence Engine).

Your job is NOT to summarize the passage.

Your job is to extract ONLY explicit, evidence-backed causal
relationships that are stated or directly supported by the
scientific passage.

Return JSON only.

============================================================
OUTPUT FORMAT
============================================================

{
  "claims": [
    {
      "src_node": "canonical_node_id",
      "dst_node": "canonical_node_id",
      "relation": "increases | decreases | enables | requires | antagonises | no_effect",
      "effect_type": "pct_change | absolute | log_response_ratio | qualitative",
      "effect_size": null,
      "effect_unit": null,
      "ci_low": null,
      "ci_high": null,
      "ci_level": null,
      "lag_years": null,
      "time_to_full_effect": null,
      "quote": "verbatim evidence quote",
      "confidence": 0.0,
      "notes": null
    }
  ]
}

If there is no valid causal claim, return:

{
  "claims": []
}

============================================================
SCIENTIFIC RULES
============================================================

1. NEVER invent scientific relationships.

2. Only extract relationships explicitly supported by the
   supplied evidence.

3. The quote must come directly from the supplied passage.

4. Use ONLY canonical node IDs from the provided ontology.

5. Do NOT create new node IDs.

6. Do NOT force a scientific concept into an approximate node.

IMPORTANT DISTINCTIONS:

- "soil water content" is NOT automatically
  "water_holding_capacity".

- "soil mineral nitrogen" is NOT automatically
  "nitrogen_fertilization".

- "soil organic carbon" is NOT automatically
  "soil organic_matter".

- "crop yield" is NOT automatically
  "agricultural production".

- "biodiversity" is NOT automatically
  "associated_biodiversity".

If the exact concept does not exist in the ontology,
DO NOT substitute a different concept.

7. If the paper reports an effect of an intervention on a
   metric, the intervention should normally be src_node and
   the metric should normally be dst_node.

8. Preserve the direction of the reported effect.

For example:

"cover cropping reduced crop yield by 7%"

becomes:

src_node = cover_crops
dst_node = crop_yield
relation = decreases
effect_type = pct_change
effect_size = -7

9. Do not reverse cause and effect.

10. If a percentage reduction is reported, use a negative
    effect_size with relation="decreases".

11. If a percentage increase is reported, use a positive
    effect_size with relation="increases".

12. For qualitative relationships with no numeric estimate,
    use effect_type="qualitative" and effect_size=null.

13. If confidence cannot reasonably be determined, use a
    conservative value rather than inventing precision.

14. Extract multiple claims when the passage explicitly
    contains multiple independent causal relationships.

15. Do not infer causality merely because two variables are
    correlated.

16. Do not turn study methods, background statements, or
    hypotheses into causal claims.

17. NEVER extract a claim from a title, section heading,
    figure caption, table caption, reference entry, or
    citation alone.

18. A relationship must have an explicit directional or
    causal statement in the supplied passage.

19. Do NOT infer the direction of an effect from the presence
    of two variables in the same sentence, title, heading,
    or caption.

20. If the passage says only that two variables were studied,
    measured, compared, associated, or investigated, but does
    not explicitly state the direction of the effect, return
    no claim.

21. Do NOT convert phrases such as "effect on", "impact on",
    "relationship between", "response of", or "associated
    with" into increases/decreases unless the passage
    explicitly gives the direction.

22. Study titles, article titles, figure captions, and table
    captions are NOT evidence of effect direction.

23. If a passage contains insufficient information to determine
    the causal direction, return:

    {"claims": []}

24. For a claim to be extracted, the quote itself must contain
    enough information to support the claimed relationship
    and direction.

25. Meta-analysis results can be extracted when the passage
    explicitly reports the effect.

26. Preserve context such as rainfall, climate, soil type,
    cropping system, or management conditions in "notes".

27. Do not infer numerical effect sizes.

28. Only provide effect_size when the passage explicitly
    reports the numerical value.

29. Confidence must reflect the strength and explicitness of
    the evidence in the supplied passage.

30. Do not extract claims from references or bibliography
    entries merely because their titles mention an intervention
    and outcome.

============================================================
ONTOLOGY
============================================================
""" + "\n" + node_text

    # --------------------------------------------------------
    # User prompt
    # --------------------------------------------------------

    user_prompt = f"""
SOURCE ID: {source_id}

Extract causal scientific claims from the following evidence
passage.

Remember:

- Use only the supplied canonical node IDs.
- Do not invent nodes.
- Do not substitute approximate concepts.
- Preserve the exact scientific meaning.
- The quote must support the claim.
- The direction must be explicitly stated.
- Do not infer direction.
- Do not extract claims from titles, headings, captions,
  references, or citations alone.
- Return JSON only.
- If there is no explicit causal relationship, return:
  {{"claims": []}}

EVIDENCE PASSAGE:
-----------------
{evidence_text}
-----------------
"""

    last_error = None

    # ========================================================
    # RETRY LOOP
    # ========================================================

    for attempt in range(1, MAX_RETRIES + 1):

        print(
            f"\nLLM extraction attempt "
            f"{attempt}/{MAX_RETRIES}..."
        )

        try:

            # ------------------------------------------------
            # Gemini invocation
            # ------------------------------------------------

            response = llm.invoke(
                [
                    (
                        "system",
                        system_prompt,
                    ),
                    (
                        "user",
                        user_prompt,
                    ),
                ]
            )

            # ------------------------------------------------
            # Normalize Gemini response
            # ------------------------------------------------

            raw_content = getattr(
                response,
                "content",
                None,
            )

            raw_text = _normalize_response_content(
                raw_content
            )

            if not raw_text.strip():
                raise ValueError(
                    "Model returned empty response."
                )

            print("\nRAW MODEL RESPONSE:")
            print(raw_text)

            # ------------------------------------------------
            # Parse JSON
            # ------------------------------------------------

            data = _extract_json(raw_text)

            # ------------------------------------------------
            # Pydantic validation
            # ------------------------------------------------

            result = ClaimExtractionResult.model_validate(
                data
            )

            # ------------------------------------------------
            # Validate individual claims
            # ------------------------------------------------

            valid_claims = []

            for claim in result.claims:

                validated = _validate_claim(
                    claim,
                    valid_nodes,
                )

                if validated is not None:
                    valid_claims.append(
                        validated
                    )

            print(
                f"VALID CLAIMS: {len(valid_claims)}"
            )

            return valid_claims

        except Exception as exc:

            last_error = exc

            error_text = str(exc)

            print(
                f"\nExtraction attempt {attempt} failed:"
            )

            print(error_text)

            # ------------------------------------------------
            # Retry with backoff
            # ------------------------------------------------

            if attempt < MAX_RETRIES:

                delay = RETRY_DELAYS[
                    min(
                        attempt - 1,
                        len(RETRY_DELAYS) - 1,
                    )
                ]

                print(
                    f"Retrying in {delay}s..."
                )

                time.sleep(delay)

    # ========================================================
    # ALL RETRIES FAILED
    # ========================================================

    raise RuntimeError(
        "LLM extraction failed after "
        f"{MAX_RETRIES} attempts: {last_error}"
    )
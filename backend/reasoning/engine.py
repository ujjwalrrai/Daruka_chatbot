from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from backend.db.database import engine
from backend.knowledge.search import search_evidence


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class CausalEffect:
    intervention: str
    metric: str
    relation: str
    effect_type: str
    effect_size: float | None
    effect_unit: str | None
    evidence_tier: str
    source_id: str
    quote: str
    confidence: float


@dataclass
class InterventionResult:
    intervention: str

    # This is GRAPH COVERAGE, NOT effectiveness.
    metric_coverage: float

    effects: list[CausalEffect]
    supporting_metrics: list[str]


# ============================================================
# DATABASE
# ============================================================

def load_causal_edges() -> list[dict[str, Any]]:
    """
    Load causal relationships from PostgreSQL.

    Only verified/provisional edges are used.
    """

    sql = text(
        """
        SELECT
            e.id,
            e.src_node,
            e.dst_node,
            e.relation,
            e.effect_type,
            e.effect_size,
            e.effect_unit,
            e.evidence_tier,
            e.source_id,
            e.quote,
            e.confidence
        FROM edge e
        WHERE e.verification_status IN ('verified', 'provisional')
        ORDER BY e.id
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(sql).mappings().all()

    return [dict(row) for row in rows]


def load_sources(
    source_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Load bibliographic metadata for the sources used by
    the causal reasoning engine.
    """

    if not source_ids:
        return {}

    sql = text(
        """
        SELECT
            id,
            title,
            authors,
            year,
            publisher,
            doi,
            url,
            evidence_tier,
            geo_scope,
            biome
        FROM source
        WHERE id = ANY(:source_ids)
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(
            sql,
            {"source_ids": source_ids},
        ).mappings().all()

    return {
        row["id"]: dict(row)
        for row in rows
    }


# ============================================================
# SITE STATE
# ============================================================

def normalize_site_state(
    site_state: Any,
) -> dict[str, Any]:
    """
    Accept:

    - Pydantic models
    - dictionaries
    - simple Python objects
    """

    if site_state is None:
        return {}

    if hasattr(site_state, "model_dump"):
        return site_state.model_dump()

    if isinstance(site_state, dict):
        return site_state

    return vars(site_state)


# ============================================================
# DEFICIENCY DETECTION
# ============================================================

def detect_deficiencies(
    site_state: Any,
) -> list[str]:
    """
    Convert observed site conditions into target metrics.

    This layer is deterministic.

    The LLM does NOT decide which environmental problems exist.
    """

    state = normalize_site_state(site_state)

    deficiencies: list[str] = []

    # --------------------------------------------------------
    # Extract possible site-state fields
    # --------------------------------------------------------

    soc = state.get("soc")

    moisture = state.get("moisture")

    # Added aliases for natural-language water conditions.
    water_retention = state.get("water_retention")
    water_holding_capacity = state.get(
        "water_holding_capacity"
    )

    rainfall = state.get("rainfall")

    land_use = state.get("land_use")

    biodiversity = state.get("biodiversity")

    erosion = state.get("erosion")

    # ========================================================
    # SOIL ORGANIC CARBON
    # ========================================================

    if soc is not None:

        if isinstance(soc, str):

            soc_text = soc.lower()

            if any(
                word in soc_text
                for word in [
                    "low",
                    "poor",
                    "depleted",
                    "declining",
                ]
            ):
                deficiencies.append(
                    "soil_organic_carbon"
                )

        elif isinstance(soc, (int, float)):

            # No universal numeric SOC threshold is assumed here.
            # Numeric interpretation should eventually come from
            # the benchmark table rather than a hard-coded value.

            pass

    # ========================================================
    # SOIL MOISTURE / WATER RETENTION
    # ========================================================

    # The site-state extractor may represent the same concept
    # using different field names.
    #
    # Examples:
    #
    # moisture = "low"
    # water_retention = "poor"
    # water_holding_capacity = "low"
    #
    # All of these map to:
    #
    # water_holding_capacity
    #
    # This keeps the final environmental metric canonical.

    water_values = [
        moisture,
        water_retention,
        water_holding_capacity,
    ]

    for water_value in water_values:

        if water_value is None:
            continue

        if isinstance(water_value, str):

            water_text = water_value.lower()

            if any(
                phrase in water_text
                for phrase in [
                    "low",
                    "dry",
                    "poor",
                    "depleted",
                    "limited",
                    "low retention",
                    "poor retention",
                    "low water retention",
                    "poor water retention",
                    "low water holding",
                    "poor water holding",
                    "low water holding capacity",
                    "poor water holding capacity",
                    "dries quickly",
                    "soil dries quickly",
                    "poor soil moisture retention",
                    "low soil moisture retention",
                ]
            ):
                deficiencies.append(
                    "water_holding_capacity"
                )

        elif isinstance(
            water_value,
            (int, float),
        ):

            # Numeric interpretation should eventually come from
            # region/soil-specific benchmarks.
            #
            # Do NOT introduce a universal numeric threshold here.

            pass

    # ========================================================
    # RAINFALL
    # ========================================================

    if rainfall is not None:

        if isinstance(
            rainfall,
            (int, float),
        ):

            # Keep the existing demo threshold.
            #
            # Later this should be replaced with a
            # region/biome-specific benchmark.

            if rainfall < 700:

                deficiencies.append(
                    "water_holding_capacity"
                )

    # ========================================================
    # LAND USE / MONOCULTURE
    # ========================================================

    if land_use is not None:

        land_use_text = str(
            land_use
        ).lower()

        if any(
            phrase in land_use_text
            for phrase in [
                "monoculture",
                "single crop",
                "monocropping",
            ]
        ):

            deficiencies.append(
                "biodiversity"
            )

    # ========================================================
    # BIODIVERSITY
    # ========================================================

    if biodiversity is not None:

        if isinstance(
            biodiversity,
            str,
        ):

            biodiversity_text = (
                biodiversity.lower()
            )

            if any(
                word in biodiversity_text
                for word in [
                    "low",
                    "declining",
                    "poor",
                ]
            ):

                deficiencies.append(
                    "biodiversity"
                )

    # ========================================================
    # SOIL EROSION
    # ========================================================

    if erosion is not None:

        if isinstance(
            erosion,
            str,
        ):

            erosion_text = erosion.lower()

            if any(
                word in erosion_text
                for word in [
                    "high",
                    "severe",
                    "increasing",
                    "poor",
                ]
            ):

                deficiencies.append(
                    "soil_erosion"
                )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    return list(
        dict.fromkeys(
            deficiencies
        )
    )


# ============================================================
# CANDIDATE DISCOVERY
# ============================================================

def find_candidate_interventions(
    deficiencies: list[str],
    edges: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Find interventions with causal relationships to
    the detected deficient metrics.
    """

    candidates: dict[
        str,
        list[dict[str, Any]]
    ] = {}

    for deficiency in deficiencies:

        for edge in edges:

            # Only consider edges that directly target
            # one of the detected deficiencies.

            if edge["dst_node"] != deficiency:
                continue

            intervention = edge[
                "src_node"
            ]

            if intervention not in candidates:

                candidates[
                    intervention
                ] = []

            candidates[
                intervention
            ].append(edge)

    return candidates


# ============================================================
# GRAPH COVERAGE
# ============================================================

def score_intervention(
    intervention: str,
    effects: list[dict[str, Any]],
    deficiencies: list[str],
) -> float:
    """
    Calculate graph coverage.

    IMPORTANT:

    This is NOT a scientific effectiveness score.

    It answers:

        "How many of the detected target metrics
        does this intervention have direct graph
        evidence for?"

    Example:

        3 detected deficiencies
        intervention affects 3

        coverage = 1.0
    """

    if not deficiencies:
        return 0.0

    target_metrics = {
        edge["dst_node"]
        for edge in effects
        if edge["dst_node"] in deficiencies
    }

    return (
        len(target_metrics)
        / len(deficiencies)
    )


# ============================================================
# BUILD RECOMMENDATIONS
# ============================================================

def build_recommendations(
    deficiencies: list[str],
    edges: list[dict[str, Any]],
) -> list[InterventionResult]:

    candidates = find_candidate_interventions(
        deficiencies,
        edges,
    )

    results: list[
        InterventionResult
    ] = []

    for intervention, effects in candidates.items():

        metric_coverage = score_intervention(
            intervention,
            effects,
            deficiencies,
        )

        causal_effects = []

        for edge in effects:

            causal_effects.append(
                CausalEffect(
                    intervention=edge[
                        "src_node"
                    ],

                    metric=edge[
                        "dst_node"
                    ],

                    relation=edge[
                        "relation"
                    ],

                    effect_type=edge[
                        "effect_type"
                    ],

                    effect_size=edge[
                        "effect_size"
                    ],

                    effect_unit=edge[
                        "effect_unit"
                    ],

                    evidence_tier=edge[
                        "evidence_tier"
                    ],

                    source_id=edge[
                        "source_id"
                    ],

                    quote=edge[
                        "quote"
                    ],

                    confidence=float(
                        edge["confidence"]
                    ),
                )
            )

        supporting_metrics = list(
            dict.fromkeys(
                effect.metric
                for effect in causal_effects
            )
        )

        results.append(
            InterventionResult(
                intervention=intervention,

                metric_coverage=metric_coverage,

                effects=causal_effects,

                supporting_metrics=supporting_metrics,
            )
        )

    # ========================================================
    # ORDERING
    # ========================================================

    # This is NOT presented as "best intervention".
    #
    # It simply places interventions with greater graph
    # coverage first so the downstream narrator can inspect
    # them.
    #
    # The API will explicitly call this "metric coverage".

    results.sort(
        key=lambda result: (
            result.metric_coverage,
            len(result.effects),
        ),
        reverse=True,
    )

    return results


# ============================================================
# EVIDENCE RETRIEVAL
# ============================================================

def retrieve_evidence(
    intervention: str,
    effects: list[CausalEffect],
    limit: int = 3,
) -> list[dict[str, Any]]:
    """
    Retrieve supporting scientific passages using pgvector.

    The query is generated from the intervention + metrics
    already present in the causal graph.
    """

    metrics = ", ".join(
        sorted(
            {
                effect.metric
                for effect in effects
            }
        )
    )

    query = (
        f"Scientific evidence about "
        f"{intervention} and its effects on "
        f"{metrics}"
    )

    rows = search_evidence(
        query,
        limit=limit,
    )

    return [
        {
            "chunk_id": row.id,

            "source_id": row.source_id,

            "page": row.page,

            "similarity": float(
                row.similarity
            ),

            "text": row.text,
        }
        for row in rows
    ]


# ============================================================
# EFFECT SUMMARY
# ============================================================

def summarize_effects(
    effects: list[CausalEffect],
) -> list[dict[str, Any]]:
    """
    Convert graph effects into a frontend-friendly structure.
    """

    output = []

    for effect in effects:

        output.append(
            {
                "metric": effect.metric,

                "relation": effect.relation,

                "effect_type": effect.effect_type,

                "effect_size": effect.effect_size,

                "effect_unit": effect.effect_unit,

                "source_id": effect.source_id,

                "evidence_tier": effect.evidence_tier,

                "confidence": effect.confidence,

                "quote": effect.quote,
            }
        )

    return output


# ============================================================
# MAIN REASONING FUNCTION
# ============================================================

def reason(
    site_state: Any,
    evidence_limit: int = 3,
) -> dict[str, Any]:
    """
    Complete CEE reasoning pipeline:

        Site state
             ↓
        Deficiency detection
             ↓
        Causal graph
             ↓
        Candidate interventions
             ↓
        Multi-metric coverage
             ↓
        Evidence retrieval
             ↓
        Source metadata
    """

    # ========================================================
    # 1. NORMALIZE SITE
    # ========================================================

    state = normalize_site_state(
        site_state
    )

    # ========================================================
    # 2. DETECT DEFICIENCIES
    # ========================================================

    deficiencies = detect_deficiencies(
        state
    )

    # ========================================================
    # 3. LOAD GRAPH
    # ========================================================

    edges = load_causal_edges()

    # ========================================================
    # 4. BUILD CANDIDATES
    # ========================================================

    recommendations = build_recommendations(
        deficiencies,
        edges,
    )

    # ========================================================
    # 5. LOAD SOURCE METADATA
    # ========================================================

    source_ids = list(
        {
            effect.source_id
            for recommendation in recommendations
            for effect in recommendation.effects
        }
    )

    sources = load_sources(
        source_ids
    )

    # ========================================================
    # 6. BUILD FINAL OUTPUT
    # ========================================================

    output = []

    for recommendation in recommendations:

        evidence = retrieve_evidence(
            recommendation.intervention,
            recommendation.effects,
            limit=evidence_limit,
        )

        recommendation_source_ids = list(
            dict.fromkeys(
                effect.source_id
                for effect in recommendation.effects
            )
        )

        recommendation_sources = [
            sources[source_id]
            for source_id in recommendation_source_ids
            if source_id in sources
        ]

        output.append(
            {
                "intervention": (
                    recommendation.intervention
                ),

                # IMPORTANT:
                # This is coverage, not effectiveness.

                "metric_coverage": (
                    recommendation.metric_coverage
                ),

                "supporting_metrics": (
                    recommendation.supporting_metrics
                ),

                "causal_effects": (
                    summarize_effects(
                        recommendation.effects
                    )
                ),

                "sources": (
                    recommendation_sources
                ),

                "retrieved_evidence": evidence,
            }
        )

    return {
        "site_state": state,

        "detected_deficiencies": (
            deficiencies
        ),

        "graph_edge_count": len(edges),

        "recommendations": output,
    }


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":

    import json

    demo_site = {
        "soc": "low",

        "rainfall": 500,

        "land_use": "monoculture",

        "water_retention": "poor",

        "biodiversity": "low",
    }

    result = reason(
        demo_site,
        evidence_limit=3,
    )

    print(
        json.dumps(
            result,
            indent=2,
            default=str,
        )
    )
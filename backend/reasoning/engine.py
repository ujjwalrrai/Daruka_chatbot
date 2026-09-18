from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from backend.db.database import engine
from backend.knowledge.search import search_source_evidence


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

    # IMPORTANT:
    # This represents graph coverage, NOT scientific effectiveness.
    metric_coverage: float

    effects: list[CausalEffect]
    supporting_metrics: list[str]


# ============================================================
# DATABASE
# ============================================================

def load_causal_edges() -> list[dict[str, Any]]:
    """
    Load causal relationships from PostgreSQL.

    Only verified and provisional edges are used.
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
            e.confidence,
            e.lag_years,
            e.time_to_full_effect
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
    Load bibliographic metadata for sources used by the
    causal reasoning engine.
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
    Normalize different site-state representations into a
    standard dictionary.

    Supports:
    - Pydantic models
    - dictionaries
    - regular Python objects
    """

    if site_state is None:
        return {}

    if hasattr(site_state, "model_dump"):
        return site_state.model_dump()

    if isinstance(site_state, dict):
        return site_state

    return vars(site_state)


# ============================================================
# CONTEXT-AWARE REASONING
# ============================================================

def derive_site_context(site_state: Any) -> dict[str, Any]:
    """
    Derive only explicit/context-safe site descriptors from the
    supplied site state.

    This function does not invent regional or soil-specific
    thresholds. It converts already observed values into simple
    labels that can be compared with explicit conditions in
    scientific evidence quotes.
    """
    state = normalize_site_state(site_state)
    context: dict[str, Any] = {}

    def text_value(key: str) -> str:
        value = state.get(key)
        return str(value).lower().strip() if value is not None else ""

    soc = text_value("soc")
    if soc:
        if any(term in soc for term in ("low", "poor", "depleted", "declining", "c-poor")):
            context["soc_context"] = "c_poor"
        elif any(term in soc for term in ("high", "rich", "c-rich")):
            context["soc_context"] = "c_rich"

    rainfall = state.get("rainfall")
    if isinstance(rainfall, (int, float)):
        # Reuse the existing demo threshold only as a descriptive
        # context label; it is not a universal scientific threshold.
        if rainfall < 700:
            context["rainfall_context"] = "lower_rainfall"
        else:
            context["rainfall_context"] = "higher_rainfall"

    land_use = text_value("land_use")
    if land_use:
        if any(term in land_use for term in ("monoculture", "single crop", "monocropping")):
            context["land_use_context"] = "monoculture"
        elif any(term in land_use for term in ("diversified", "diversification", "mixed cropping", "intercropping")):
            context["land_use_context"] = "diversified"

    biodiversity = text_value("biodiversity")
    if biodiversity:
        if any(term in biodiversity for term in ("low", "poor", "declining")):
            context["biodiversity_context"] = "low"
        elif any(term in biodiversity for term in ("high", "rich", "good")):
            context["biodiversity_context"] = "high"

    # Preserve explicitly supplied categorical context fields when present.
    for key in ("region", "biome", "soil_type", "climate", "soil_texture"):
        value = state.get(key)
        if value is not None and str(value).strip():
            context[key] = str(value).strip().lower()

    return context


def assess_edge_context(
    edge: dict[str, Any],
    site_state: Any,
) -> dict[str, Any]:
    """
    Compare explicit conditions in an evidence quote with the site.

    Rules are deliberately conservative:
    - explicit agreement -> matched
    - explicit contradiction -> rejected
    - no explicit condition -> neutral

    We never infer that an intervention is effective merely because
    the site happens to have a similar condition.
    """
    context = derive_site_context(site_state)
    quote = str(edge.get("quote") or "").lower()
    reasons: list[str] = []
    contradictions: list[str] = []
    matches: list[str] = []

    # Soil-carbon context. This is particularly important for S09,
    # where some N-fertilization effects differ between C-poor and
    # C-rich soils.
    soc_context = context.get("soc_context")
    mentions_c_poor = any(term in quote for term in (
        "c-poor", "c poor", "c-poor soils", "c-poor soil",
        "low-carbon soil", "low carbon soil", "low soil carbon",
        "carbon-poor", "carbon poor",
    ))
    mentions_c_rich = any(term in quote for term in (
        "c-rich", "c rich", "c-rich soils", "c-rich soil",
        "high-carbon soil", "high carbon soil", "high soil carbon",
        "carbon-rich", "carbon rich",
    ))

    if soc_context == "c_poor":
        if mentions_c_rich and not mentions_c_poor:
            contradictions.append("evidence is explicitly for C-rich/high-carbon soils")
        elif mentions_c_poor:
            matches.append("site has low/C-poor soil carbon and evidence is for C-poor soils")
    elif soc_context == "c_rich":
        if mentions_c_poor and not mentions_c_rich:
            contradictions.append("evidence is explicitly for C-poor/low-carbon soils")
        elif mentions_c_rich:
            matches.append("site has high/C-rich soil carbon and evidence is for C-rich soils")

    # Climate / rainfall context. Only explicit dry/wet terminology is
    # treated as scientific context; the rainfall value itself is not
    # used to claim that a paper applies to a climate unless the quote
    # states such a condition.
    rainfall_context = context.get("rainfall_context")
    dry_terms = ("dryland", "dry land", "semi-arid", "semi arid", "arid", "low rainfall", "low-rainfall")
    wet_terms = ("humid", "high rainfall", "high-rainfall", "wet climate", "wetland")
    mentions_dry = any(term in quote for term in dry_terms)
    mentions_wet = any(term in quote for term in wet_terms)

    if rainfall_context == "lower_rainfall":
        if mentions_wet and not mentions_dry:
            contradictions.append("evidence is explicitly for a wetter/high-rainfall context")
        elif mentions_dry:
            matches.append("site has lower rainfall and evidence is explicitly from a dry/semi-arid context")
    elif rainfall_context == "higher_rainfall":
        if mentions_dry and not mentions_wet:
            contradictions.append("evidence is explicitly for a dry/semi-arid context")
        elif mentions_wet:
            matches.append("site has higher rainfall and evidence is explicitly from a wetter context")

    # Region/biome/soil-type exact phrase matching. These are only
    # positive context signals; absence of a match remains neutral.
    for key in ("region", "biome", "soil_type", "climate", "soil_texture"):
        value = context.get(key)
        if not value:
            continue
        normalized = str(value).replace("_", " ").lower()
        if normalized in quote:
            matches.append(f"{key} '{normalized}' is explicitly mentioned in the evidence")

    if contradictions:
        status = "contradicted"
        score = 0.0
        reasons.extend(contradictions)
    elif matches:
        status = "matched"
        score = 1.0
        reasons.extend(matches)
    else:
        status = "neutral"
        score = 0.5
        reasons.append("no explicit site-context condition was found in the evidence quote")

    return {
        "status": status,
        "score": score,
        "reasons": reasons,
        "matches": matches,
        "contradictions": contradictions,
    }


def filter_edges_by_context(
    edges: list[dict[str, Any]],
    site_state: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Remove only causally relevant edges whose own evidence explicitly
    contradicts the observed site context.

    Returns:
        usable_edges, context_audit
    """
    usable_edges: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []

    for edge in edges:
        assessment = assess_edge_context(edge, site_state)
        enriched = dict(edge)
        enriched["context_status"] = assessment["status"]
        enriched["context_score"] = assessment["score"]
        enriched["context_reasons"] = assessment["reasons"]

        audit.append({
            "edge_id": edge.get("id"),
            "source": edge.get("src_node"),
            "target": edge.get("dst_node"),
            "source_id": edge.get("source_id"),
            "status": assessment["status"],
            "score": assessment["score"],
            "reasons": assessment["reasons"],
        })

        if assessment["status"] != "contradicted":
            usable_edges.append(enriched)

    return usable_edges, audit


# ============================================================
# DEFICIENCY DETECTION
# ============================================================

def detect_deficiencies(
    site_state: Any,
) -> list[str]:
    """
    Convert observed site conditions into canonical environmental
    metrics.

    This layer is deterministic.

    The LLM is NOT responsible for deciding which environmental
    deficiencies exist.
    """

    state = normalize_site_state(site_state)

    deficiencies: list[str] = []

    # --------------------------------------------------------
    # Extract site-state fields
    # --------------------------------------------------------

    soc = state.get("soc")

    moisture = state.get("moisture")

    water_retention = state.get(
        "water_retention"
    )

    water_holding_capacity = state.get(
        "water_holding_capacity"
    )

    rainfall = state.get(
        "rainfall"
    )

    land_use = state.get(
        "land_use"
    )

    biodiversity = state.get(
        "biodiversity"
    )

    erosion = state.get(
        "erosion"
    )

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

        elif isinstance(
            soc,
            (int, float),
        ):

            # Numeric SOC thresholds should eventually come from
            # region/soil-specific benchmark data.
            #
            # We intentionally do not assume a universal numeric
            # threshold here.

            pass

    # ========================================================
    # WATER / MOISTURE
    # ========================================================

    water_values = [
        moisture,
        water_retention,
        water_holding_capacity,
    ]

    for water_value in water_values:

        if water_value is None:
            continue

        if isinstance(
            water_value,
            str,
        ):

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

            # Numeric interpretation should eventually use
            # benchmark data rather than a universal threshold.

            pass

    # ========================================================
    # RAINFALL
    # ========================================================

    if rainfall is not None:

        if isinstance(
            rainfall,
            (int, float),
        ):

            # Existing demo threshold.
            #
            # This should eventually become region/biome-specific.

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
    Find interventions that have direct causal relationships
    to the detected deficient metrics.
    """

    candidates: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for deficiency in deficiencies:

        for edge in edges:

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
    Calculate how many detected environmental deficiencies
    are directly covered by this intervention in the causal graph.

    IMPORTANT:

    This is NOT a scientific effectiveness score.

    Example:

        3 detected deficiencies
        3 directly covered

        metric_coverage = 1.0
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
    """
    Construct intervention candidates and their causal effects.
    """

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

        causal_effects: list[
            CausalEffect
        ] = []

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

    # --------------------------------------------------------
    # Sort by graph coverage.
    #
    # This is NOT a "best intervention" ranking.
    # It simply makes higher-coverage graph candidates appear
    # first for downstream processing.
    # --------------------------------------------------------

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
    Retrieve scientific evidence for the selected causal edges.

    IMPORTANT PROVENANCE RULE:

    Evidence retrieval is restricted to the source attached
    to each causal graph edge.

    This prevents the semantic search engine from returning
    scientifically unrelated chunks merely because their
    wording happens to be semantically similar.

    Query construction uses natural scientific terminology
    instead of internal graph identifiers.
    """

    # ========================================================
    # INTERNAL METRIC → SCIENTIFIC TERMINOLOGY
    # ========================================================

    terminology = {

        "soil_organic_carbon": (
            "soil organic carbon SOC "
            "soil carbon"
        ),

        "soil_organic_matter": (
            "soil organic matter "
            "soil organic matter content"
        ),

        "water_holding_capacity": (
            "soil water holding capacity "
            "water retention "
            "soil moisture"
        ),

        "biodiversity": (
            "biodiversity "
            "species diversity "
            "ecosystem diversity"
        ),

        "associated_biodiversity": (
            "associated biodiversity "
            "species diversity "
            "farmland biodiversity"
        ),

        "agricultural_production": (
            "agricultural production "
            "crop production"
        ),

        "crop_yield": (
            "crop yield "
            "crop productivity"
        ),

        "pest_disease_control": (
            "pest and disease control "
            "pest suppression "
            "disease suppression"
        ),

        "carbon_sequestration": (
            "carbon sequestration "
            "soil carbon sequestration"
        ),

        "soil_quality": (
            "soil quality "
            "soil health"
        ),

        "soil_erosion": (
            "soil erosion "
            "erosion control"
        ),

        "plant_biomass": (
            "plant biomass "
            "vegetation biomass"
        ),

        "crop_diversification": (
            "crop diversification "
            "diversified cropping systems"
        ),
    }

    # ========================================================
    # INTERVENTION → SCIENTIFIC TERMINOLOGY
    # ========================================================

    intervention_terms = {

        "conservation_agriculture": (
            "conservation agriculture "
            "minimum tillage "
            "reduced tillage "
            "residue retention "
            "soil cover"
        ),

        "cover_crops": (
            "cover crops "
            "cover cropping "
            "living cover crops"
        ),

        "crop_rotation": (
            "crop rotation "
            "crop rotations"
        ),

        "crop_diversification": (
            "crop diversification "
            "diversified cropping systems"
        ),

        "intercropping": (
            "intercropping "
            "intercrop systems"
        ),

        "agroforestry": (
            "agroforestry "
            "trees and crops "
            "tree-based agricultural systems"
        ),

        "manure_application": (
            "manure application "
            "animal manure "
            "organic amendments"
        ),

        "minimum_tillage": (
            "minimum tillage "
            "reduced tillage "
            "conservation tillage"
        ),

        "residue_retention": (
            "crop residue retention "
            "residue management"
        ),

        "soil_cover": (
            "soil cover "
            "ground cover "
            "vegetative cover"
        ),

        "variety_mixtures": (
            "variety mixtures "
            "cultivar mixtures "
            "crop genetic diversity"
        ),

        "n_fertilization": (
            "nitrogen fertilization "
            "nitrogen fertilizer"
        ),
    }

    # ========================================================
    # RELATION → SCIENTIFIC LANGUAGE
    # ========================================================

    relation_terms = {

        "increases": (
            "increase "
            "improve "
            "enhance "
            "higher "
            "greater"
        ),

        "decreases": (
            "decrease "
            "reduce "
            "lower "
            "reduced"
        ),

        "improves": (
            "improve "
            "enhance "
            "increase"
        ),

        "reduces": (
            "reduce "
            "decrease "
            "lower"
        ),

        "supports": (
            "support "
            "promote "
            "benefit"
        ),
    }

    evidence: list[
        dict[str, Any]
    ] = []

    # ========================================================
    # SEARCH EACH CAUSAL EFFECT
    # ========================================================

    for effect in effects:

        intervention_text = intervention_terms.get(
            intervention,
            intervention.replace(
                "_",
                " ",
            ),
        )

        metric_text = terminology.get(
            effect.metric,
            effect.metric.replace(
                "_",
                " ",
            ),
        )

        relation_text = relation_terms.get(
            effect.relation,
            effect.relation.replace(
                "_",
                " ",
            ),
        )

        # ----------------------------------------------------
        # Natural-language semantic query
        # ----------------------------------------------------

        query = (
            f"Research evidence on "
            f"{intervention_text}. "
            f"Studies measuring "
            f"{metric_text}. "
            f"Reported effects include "
            f"{relation_text}."
        )

        # ----------------------------------------------------
        # CRITICAL:
        #
        # Restrict search to the source that supports
        # this causal edge.
        # ----------------------------------------------------

        rows = search_source_evidence(
            query=query,
            source_id=effect.source_id,
            intervention=intervention,
            metric=effect.metric,
            relation=effect.relation,
            limit=limit,
        )

        for row in rows:

            evidence.append(
                {
                    "chunk_id": row.id,

                    "source_id": row.source_id,

                    "page": row.page,

                    "section": row.section,

                    "similarity": float(
                        row.similarity
                    ),

                    "text": row.text,

                    # ------------------------------------------------
                    # Provenance metadata
                    # ------------------------------------------------

                    "intervention": intervention,

                    "metric": effect.metric,

                    "relation": effect.relation,

                    # Source supporting the causal graph edge.
                    "causal_source_id": effect.source_id,

                    # Evidence provenance / quality from the retriever.
                    "claim_provenance": getattr(
                        row,
                        "claim_provenance",
                        "text_match",
                    ),

                    "evidence_quality": getattr(
                        row,
                        "evidence_quality",
                        "text_match",
                    ),
                }
            )

    # ========================================================
    # DEDUPLICATE
    # ========================================================

    unique_evidence: list[
        dict[str, Any]
    ] = []

    seen_evidence: set[tuple] = set()

    for item in evidence:

        dedup_key = (
            item["source_id"],
            " ".join(
                str(item["text"] or "").lower().split()
            ),
        )

        if dedup_key in seen_evidence:
            continue

        seen_evidence.add(
            dedup_key
        )

        unique_evidence.append(
            item
        )

    # ========================================================
    # SORT BY SEMANTIC SIMILARITY
    # ========================================================

    unique_evidence.sort(
        key=lambda item: item[
            "similarity"
        ],
        reverse=True,
    )

    # ========================================================
    # FINAL LIMIT
    # ========================================================

    return unique_evidence[
        :limit
    ]


# ============================================================
# MECHANISM-AWARE REASONING
# ============================================================

def load_mechanism_nodes() -> set[str]:
    """Load mechanism nodes dynamically from the graph."""
    sql = text(
        """
        SELECT canonical_name
        FROM node
        WHERE kind::text = 'mechanism'
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(sql).mappings().all()

    return {row["canonical_name"] for row in rows}


def retrieve_mechanism_paths(
    intervention: str,
    edges: list[dict[str, Any]],
    max_depth: int = 3,
    min_confidence: float = 0.50,
) -> list[dict[str, Any]]:
    """Traverse the causal graph through known mechanisms."""
    mechanism_nodes = load_mechanism_nodes()

    adjacency: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        confidence = float(edge["confidence"])
        if confidence < min_confidence:
            continue
        adjacency.setdefault(edge["src_node"], []).append(edge)

    paths: list[dict[str, Any]] = []

    def walk(current_node: str, path_nodes: list[str], path_edges: list[dict[str, Any]]) -> None:
        if len(path_edges) >= max_depth:
            return
        for edge in adjacency.get(current_node, []):
            next_node = edge["dst_node"]
            if next_node in path_nodes:
                continue

            new_nodes = path_nodes + [next_node]
            new_edges = path_edges + [edge]
            mechanisms = [node for node in new_nodes if node in mechanism_nodes]

            if mechanisms:
                paths.append({
                    "intervention": intervention,
                    "nodes": new_nodes,
                    "mechanisms": mechanisms,
                    "final_metric": next_node,
                    "path_confidence": min(float(e["confidence"]) for e in new_edges),
                    "edges": [
                        {
                            "source": e["src_node"],
                            "target": e["dst_node"],
                            "relation": e["relation"],
                            "effect_type": e["effect_type"],
                            "effect_size": e["effect_size"],
                            "effect_unit": e["effect_unit"],
                            "source_id": e["source_id"],
                            "confidence": float(e["confidence"]),
                            "quote": e["quote"],
                        }
                        for e in new_edges
                    ],
                })

            walk(next_node, new_nodes, new_edges)

    walk(intervention, [intervention], [])

    unique_paths: dict[tuple[str, ...], dict[str, Any]] = {}
    for path in paths:
        key = tuple(path["nodes"])
        existing = unique_paths.get(key)
        if existing is None or path["path_confidence"] > existing["path_confidence"]:
            unique_paths[key] = path

    return sorted(
        unique_paths.values(),
        key=lambda item: (
            item["path_confidence"],
            len(item["mechanisms"]),
            -len(item["nodes"]),
        ),
        reverse=True,
    )


def calculate_path_confidence(path: dict[str, Any]) -> float:
    """Propagate confidence conservatively using the weakest edge."""
    confidences = [
        float(edge["confidence"])
        for edge in path.get("edges", [])
        if edge.get("confidence") is not None
    ]
    return min(confidences) if confidences else 0.0


def summarize_time_horizon(
    effects: list[CausalEffect],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize temporal metadata without inventing missing values."""
    if not effects:
        return {
            "available": False,
            "min_lag_years": None,
            "max_lag_years": None,
            "min_time_to_full_effect_years": None,
            "max_time_to_full_effect_years": None,
            "description": "No causal effects available.",
        }

    intervention = effects[0].intervention
    direct_edges = [
        edge for edge in edges
        if edge.get("src_node") == intervention
    ]

    lag_years = [
        float(edge["lag_years"])
        for edge in direct_edges
        if edge.get("lag_years") is not None
    ]
    full_effect_years = [
        float(edge["time_to_full_effect"])
        for edge in direct_edges
        if edge.get("time_to_full_effect") is not None
    ]

    available = bool(lag_years or full_effect_years)
    if not available:
        description = "Temporal metadata is not available for the direct causal edges."
    else:
        parts = []
        if lag_years:
            parts.append(f"observed lag: {min(lag_years):g}–{max(lag_years):g} years")
        if full_effect_years:
            parts.append(f"time to full effect: {min(full_effect_years):g}–{max(full_effect_years):g} years")
        description = "; ".join(parts) + "."

    return {
        "available": available,
        "min_lag_years": min(lag_years) if lag_years else None,
        "max_lag_years": max(lag_years) if lag_years else None,
        "min_time_to_full_effect_years": min(full_effect_years) if full_effect_years else None,
        "max_time_to_full_effect_years": max(full_effect_years) if full_effect_years else None,
        "description": description,
    }


# ============================================================
# EFFECT SUMMARY
# ============================================================

def summarize_effects(
    effects: list[CausalEffect],
) -> list[dict[str, Any]]:
    """
    Convert causal effects into a JSON-friendly structure.
    """

    output: list[
        dict[str, Any]
    ] = []

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
        Multi-metric graph coverage
             ↓
        Source-restricted evidence retrieval
             ↓
        Source metadata
             ↓
        Structured result
    """

    # ========================================================
    # 1. NORMALIZE SITE STATE
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
    # 3. LOAD CAUSAL GRAPH
    # ========================================================

    edges = load_causal_edges()

    # ========================================================
    # 4. CONTEXT-AWARE GRAPH FILTERING
    # ========================================================

    context_aware_edges, context_audit = filter_edges_by_context(
        edges,
        state,
    )

    # Only explicitly contradicted evidence is removed. Neutral
    # evidence remains available so that missing context never
    # becomes an unsupported rejection.
    
    # ========================================================
    # 5. BUILD INTERVENTION CANDIDATES
    # ========================================================

    recommendations = build_recommendations(
        deficiencies,
        context_aware_edges,
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
    # 6. BUILD FINAL RECOMMENDATION OUTPUT
    # ========================================================

    output: list[
        dict[str, Any]
    ] = []

    for recommendation in recommendations:

    # ----------------------------------------------------
    # Retrieve evidence using the dedicated evidence
    # retrieval pipeline.
    #
    # This performs:
    # - source restriction
    # - causal-edge provenance checking
    # - evidence-quality classification
    # - duplicate removal
    # - semantic ranking
    # ----------------------------------------------------

        evidence = retrieve_evidence(
            intervention=recommendation.intervention,
            effects=recommendation.effects,
            limit=evidence_limit,
    )

        mechanism_paths = retrieve_mechanism_paths(
            intervention=recommendation.intervention,
            edges=context_aware_edges,
            max_depth=3,
            min_confidence=0.50,
        )

        for path in mechanism_paths:
            path["path_confidence"] = calculate_path_confidence(path)

        time_horizon = summarize_time_horizon(
            effects=recommendation.effects,
            edges=context_aware_edges,
        )

        recommendation_source_ids = list(
            dict.fromkeys(
                effect.source_id
                for effect in recommendation.effects
            )
        )

        recommendation_sources = [
            sources[source_id]
            for source_id
            in recommendation_source_ids
            if source_id in sources
        ]

        recommendation_context = [
            item
            for item in context_audit
            if item["source"] == recommendation.intervention
            or any(
                effect.source_id == item["source_id"]
                and effect.metric == item["target"]
                for effect in recommendation.effects
            )
        ]

        output.append(
            {
                "intervention": (
                    recommendation.intervention
                ),

                "context_assessment": {
                    "site_context": derive_site_context(state),
                    "filtered_edge_count": len(context_aware_edges),
                    "context_edges_rejected": sum(
                        1
                        for item in recommendation_context
                        if item["status"] == "contradicted"
                    ),
                    "edge_audit": recommendation_context,
                },

                # Graph coverage only.
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

                "mechanism_paths": mechanism_paths,

                "time_horizon": time_horizon,
            }
        )

    # ========================================================
    # RETURN COMPLETE ANALYSIS
    # ========================================================

    return {
        "site_state": state,

        "detected_deficiencies": (
            deficiencies
        ),

        "graph_edge_count": len(
            edges
        ),

        "context_aware_graph_edge_count": len(
            context_aware_edges
        ),

        "context_summary": {
            "site_context": derive_site_context(state),
            "rejected_edge_count": sum(
                1
                for item in context_audit
                if item["status"] == "contradicted"
            ),
            "rejected_edges": [
                item
                for item in context_audit
                if item["status"] == "contradicted"
            ],
        },

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
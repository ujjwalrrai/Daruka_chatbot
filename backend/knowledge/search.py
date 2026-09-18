from __future__ import annotations

import re

from sqlalchemy import text

from backend.db.database import engine
from backend.knowledge.embeddings import embed_texts


# ============================================================
# SCIENTIFIC TERMINOLOGY
# ============================================================

INTERVENTION_TERMS = {
    "conservation_agriculture": [
        "conservation agriculture",
        "conservation farming",
        "minimum tillage",
        "reduced tillage",
        "no tillage",
        "zero tillage",
        "residue retention",
        "soil cover",
    ],
    "cover_crops": [
        "cover crops",
        "cover crop",
        "cover cropping",
    ],
    "crop_rotation": [
        "crop rotation",
        "crop rotations",
        "rotational cropping",
    ],
    "crop_diversification": [
        "crop diversification",
        "diversified cropping",
        "diversified cropping systems",
    ],
    "intercropping": [
        "intercropping",
        "intercrop",
        "intercropped",
    ],
    "agroforestry": [
        "agroforestry",
        "agroforestry systems",
        "trees and crops",
    ],
    "manure_application": [
        "manure application",
        "animal manure",
        "manure",
        "organic amendment",
        "organic amendments",
    ],
    "minimum_tillage": [
        "minimum tillage",
        "reduced tillage",
        "conservation tillage",
        "low tillage",
    ],
    "residue_retention": [
        "residue retention",
        "crop residue",
        "residue management",
        "crop residues",
    ],
    "soil_cover": [
        "soil cover",
        "ground cover",
        "vegetative cover",
    ],
    "variety_mixtures": [
        "variety mixtures",
        "cultivar mixtures",
        "varietal mixtures",
        "genetic diversity",
    ],
    "n_fertilization": [
        "nitrogen fertilization",
        "nitrogen fertilizer",
        "nitrogen application",
        "n fertilization",
    ],
}


METRIC_TERMS = {
    "soil_organic_carbon": [
        "soil organic carbon",
        "soil organic c",
        "soil carbon",
        "soc",
    ],
    "soil_organic_matter": [
        "soil organic matter",
        "organic matter",
        "som",
    ],
    "water_holding_capacity": [
        "water holding capacity",
        "water-holding capacity",
        "water retention",
        "soil water retention",
        "soil moisture retention",
        "water availability",
        "available water",
    ],
    "biodiversity": [
        "biodiversity",
        "species diversity",
        "species richness",
        "ecosystem diversity",
        "biological diversity",
    ],
    "associated_biodiversity": [
        "associated biodiversity",
        "farmland biodiversity",
        "species diversity",
        "species richness",
    ],
    "agricultural_production": [
        "agricultural production",
        "crop production",
        "farm production",
        "productivity",
    ],
    "crop_yield": [
        "crop yield",
        "crop yields",
        "yield",
        "productivity",
    ],
    "pest_disease_control": [
        "pest control",
        "pest suppression",
        "disease control",
        "disease suppression",
        "pest and disease control",
    ],
    "carbon_sequestration": [
        "carbon sequestration",
        "soil carbon sequestration",
        "carbon storage",
        "soil carbon storage",
    ],
    "soil_quality": [
        "soil quality",
        "soil health",
    ],
    "soil_erosion": [
        "soil erosion",
        "erosion",
        "erosion control",
    ],
    "plant_biomass": [
        "plant biomass",
        "vegetation biomass",
        "biomass production",
    ],
    "labour_requirement": [
        "labour requirement",
        "labor requirement",
        "labour",
        "labor",
    ],
}


RELATION_TERMS = {
    "increases": [
        "increase",
        "increases",
        "increased",
        "improve",
        "improves",
        "improved",
        "enhance",
        "enhances",
        "enhanced",
        "higher",
        "greater",
    ],
    "decreases": [
        "decrease",
        "decreases",
        "decreased",
        "reduce",
        "reduces",
        "reduced",
        "lower",
        "lowers",
        "less",
    ],
    "improves": [
        "improve",
        "improves",
        "improved",
        "enhance",
        "enhances",
        "enhanced",
        "increase",
        "increases",
    ],
    "reduces": [
        "reduce",
        "reduces",
        "reduced",
        "decrease",
        "decreases",
        "decreased",
        "lower",
        "lowers",
    ],
    "supports": [
        "support",
        "supports",
        "promote",
        "promotes",
        "benefit",
        "benefits",
    ],
}


# ============================================================
# HELPERS
# ============================================================

def get_terms(
    mapping: dict[str, list[str]],
    key: str,
) -> list[str]:
    """Return scientific terminology for a canonical graph term."""

    terms = mapping.get(key)

    if terms:
        return terms

    return [
        key.replace("_", " ")
    ]


def normalize_text(value: str) -> str:
    """Normalize text for lexical matching."""

    value = value.lower()
    value = value.replace("-", " ")

    value = re.sub(
        r"[^a-z0-9\s]",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


# ============================================================
# EDGE-LEVEL PROVENANCE
# ============================================================

_EDGE_SECTION_RE = re.compile(
    r"edge\s+\d+\s*:\s*([a-z0-9_ -]+?)\s*(?:->|→)\s*([a-z0-9_ -]+)",
    re.IGNORECASE,
)


def _canonicalize_edge_endpoint(value: str) -> str:
    """Normalize an edge endpoint to the canonical graph node format."""
    value = value.strip().lower()
    value = re.sub(r"[-\s]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def parse_edge_provenance(section: str | None) -> tuple[str, str] | None:
    """
    Parse explicit causal-edge provenance from evidence sections.

    Examples:
        Edge 4: cover_crops -> soil_organic_carbon
        Edge 65: conservation_agriculture -> water_holding_capacity

    Returns:
        (source_node, target_node), or None if the section does not
        contain an explicit edge signature.
    """
    if not section:
        return None

    match = _EDGE_SECTION_RE.search(section)
    if not match:
        return None

    source = _canonicalize_edge_endpoint(match.group(1))
    target = _canonicalize_edge_endpoint(match.group(2))

    if not source or not target:
        return None

    return source, target


def edge_provenance_matches(
    section: str | None,
    intervention: str,
    metric: str,
) -> bool | None:
    """
    Compare explicit section-level edge provenance with the requested edge.

    Returns:
        True  = explicit edge matches requested intervention -> metric
        False = explicit edge exists but is a different edge
        None  = no explicit edge provenance in the section
    """
    provenance = parse_edge_provenance(section)

    if provenance is None:
        return None

    source, target = provenance

    requested_source = _canonicalize_edge_endpoint(intervention)
    requested_target = _canonicalize_edge_endpoint(metric)

    return source == requested_source and target == requested_target


def term_matches(
    text_value: str,
    terms: list[str],
) -> list[str]:
    """Return terminology actually present in evidence text."""

    normalized = normalize_text(
        text_value
    )

    matches = []

    for term in terms:

        normalized_term = normalize_text(
            term
        )

        if normalized_term in normalized:
            matches.append(term)

    return matches


# ============================================================
# SEMANTIC QUERY
# ============================================================

def build_search_query(
    query: str,
    intervention: str | None = None,
    metric: str | None = None,
    relation: str | None = None,
) -> str:
    """
    Build a natural-language semantic query.
    """

    parts = [query]

    if intervention:
        parts.append(
            "Intervention: "
            + ", ".join(
                get_terms(
                    INTERVENTION_TERMS,
                    intervention,
                )[:5]
            )
        )

    if metric:
        parts.append(
            "Metric: "
            + ", ".join(
                get_terms(
                    METRIC_TERMS,
                    metric,
                )[:5]
            )
        )

    if relation:
        parts.append(
            "Effect: "
            + ", ".join(
                get_terms(
                    RELATION_TERMS,
                    relation,
                )[:5]
            )
        )

    return ". ".join(parts)


# ============================================================
# CAUSAL RELEVANCE
# ============================================================

def calculate_causal_relevance(
    text_value: str,
    intervention: str | None = None,
    metric: str | None = None,
    relation: str | None = None,
) -> tuple[float, dict]:
    """
    Calculate lexical relevance to the requested causal edge.

    Components:

        Intervention = 0.40
        Metric       = 0.45
        Relation     = 0.15

    This score is separate from vector similarity.
    """

    intervention_terms = (
        get_terms(
            INTERVENTION_TERMS,
            intervention,
        )
        if intervention
        else []
    )

    metric_terms = (
        get_terms(
            METRIC_TERMS,
            metric,
        )
        if metric
        else []
    )

    relation_terms = (
        get_terms(
            RELATION_TERMS,
            relation,
        )
        if relation
        else []
    )

    intervention_matches = term_matches(
        text_value,
        intervention_terms,
    )

    metric_matches = term_matches(
        text_value,
        metric_terms,
    )

    relation_matches = term_matches(
        text_value,
        relation_terms,
    )

    intervention_score = (
        1.0
        if intervention_matches
        else 0.0
    )

    metric_score = (
        1.0
        if metric_matches
        else 0.0
    )

    relation_score = (
        1.0
        if relation_matches
        else 0.0
    )

    score = (
        0.40 * intervention_score
        + 0.45 * metric_score
        + 0.15 * relation_score
    )

    return score, {
        "intervention_matches": intervention_matches,
        "metric_matches": metric_matches,
        "relation_matches": relation_matches,
    }


# ============================================================
# EVIDENCE QUALITY + DEDUPLICATION
# ============================================================

def classify_evidence_quality(
    text_value: str,
    claim_provenance: str,
    intervention: str | None = None,
) -> str:
    """
    Classify evidence without changing the scientific claim.

    direct_edge:
        The chunk explicitly belongs to the requested causal edge.

    multi_intervention_synthesis:
        The evidence explicitly discusses multiple management
        practices/interventions in the same synthesis statement.

    text_match:
        No explicit edge provenance; retained through lexical/vector
        fallback.
    """
    if claim_provenance != "direct_edge":
        return "text_match"

    if not intervention:
        return "direct_edge"

    normalized = normalize_text(text_value)

    # Count distinct canonical intervention concepts that are explicitly
    # represented in the evidence text.
    matched_interventions = 0
    for canonical, terms in INTERVENTION_TERMS.items():
        if term_matches(normalized, terms):
            matched_interventions += 1

    if matched_interventions > 1:
        return "multi_intervention_synthesis"

    return "direct_edge"


def evidence_dedup_key(row) -> tuple:
    """
    Deduplicate repeated evidence chunks.

    Prefer the actual evidence text + source + section so that repeated
    database rows carrying the same scientific statement collapse into
    one result while genuinely different claims remain separate.
    """
    text_key = normalize_text(row.text or "")
    section_key = normalize_text(row.section or "")

    return (
        str(row.source_id),
        text_key,
    )


# ============================================================
# RESULT OBJECT
# ============================================================

class EvidenceSearchResult:
    """
    Result object compatible with engine.py.

    Available attributes:

        id
        source_id
        page
        section
        text
        similarity
        vector_similarity
        causal_relevance
        claim_provenance
        evidence_quality
    """

    def __init__(
        self,
        id,
        source_id,
        page,
        section,
        text,
        similarity,
        vector_similarity,
        causal_relevance,
        claim_provenance="text_match",
        evidence_quality="text_match",
    ):
        self.id = id
        self.source_id = source_id
        self.page = page
        self.section = section
        self.text = text
        self.similarity = similarity
        self.vector_similarity = vector_similarity
        self.causal_relevance = causal_relevance
        self.claim_provenance = claim_provenance
        self.evidence_quality = evidence_quality


# ============================================================
# EVIDENCE SEARCH
# ============================================================

def search_evidence(
    query: str,
    limit: int = 5,
    source_id: str | None = None,
    intervention: str | None = None,
    metric: str | None = None,
    relation: str | None = None,
):
    """
    Hybrid evidence retrieval.

    Pipeline:

        Vector similarity
                +
        Intervention text match
                +
        Metric text match
                +
        Relation text match
                ↓
        Causal relevance
                ↓
        Final ranking

    source_id is a HARD provenance filter.
    """

    # --------------------------------------------------------
    # Build semantic query
    # --------------------------------------------------------

    search_query = build_search_query(
        query=query,
        intervention=intervention,
        metric=metric,
        relation=relation,
    )

    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    embedding = embed_texts(
        [search_query]
    )[0]

    # --------------------------------------------------------
    # Retrieve candidates
    # --------------------------------------------------------

    candidate_limit = max(
        limit * 10,
        30,
    )

    params = {
        "embedding": str(embedding),
        "candidate_limit": candidate_limit,
    }

    # --------------------------------------------------------
    # Source provenance filter
    # --------------------------------------------------------

    source_filter = ""

    if source_id:

        source_filter = """
            AND ec.source_id = :source_id
        """

        params["source_id"] = source_id

    # --------------------------------------------------------
    # pgvector search
    # --------------------------------------------------------

    sql = text(
        f"""
        SELECT
            ec.id,
            ec.source_id,
            ec.page,
            ec.section,
            ec.text,

            1 - (
                ec.embedding
                <=> CAST(
                    :embedding AS vector
                )
            ) AS vector_similarity

        FROM evidence_chunk ec

        WHERE
            ec.embedding IS NOT NULL

            {source_filter}

            AND LENGTH(
                TRIM(ec.text)
            ) >= 250

            AND ec.text NOT LIKE
                'Productivity limits%'

            AND ec.text NOT LIKE
                'Cover crop impacts%'

            AND ec.text NOT LIKE
                'Animal manure application%'

            AND ec.text NOT LIKE
                'Roots contribute%'

        ORDER BY
            ec.embedding
            <=> CAST(
                :embedding AS vector
            )

        LIMIT :candidate_limit
        """
    )

    with engine.connect() as connection:

        rows = connection.execute(
            sql,
            params,
        ).fetchall()

    # --------------------------------------------------------
    # Hybrid ranking
    # --------------------------------------------------------

    ranked = []

    for row in rows:

        # --------------------------------------------------------
        # HARD EDGE-LEVEL PROVENANCE FILTER
        # --------------------------------------------------------
        #
        # If a chunk explicitly identifies itself as:
        #
        #     Edge 4: cover_crops -> soil_organic_carbon
        #
        # it cannot be used as direct evidence for:
        #
        #     manure_application -> soil_organic_carbon
        #
        # even when the text happens to mention manure as well.
        #
        # Chunks without an explicit Edge X: A -> B section continue
        # through the normal hybrid retrieval fallback.
        provenance_match = (
            edge_provenance_matches(
                section=row.section,
                intervention=intervention,
                metric=metric,
            )
            if intervention and metric
            else None
        )

        if provenance_match is False:
            continue

        causal_score, matches = (
            calculate_causal_relevance(
                row.text,
                intervention=intervention,
                metric=metric,
                relation=relation,
            )
        )

        vector_similarity = float(
            row.vector_similarity
        )

        if (
            intervention
            or metric
            or relation
        ):
            hybrid_score = (
                0.30 * vector_similarity
                + 0.70 * causal_score
            )
        else:
            hybrid_score = vector_similarity

        claim_provenance = (
            "direct_edge"
            if provenance_match is True
            else "text_match"
        )

        evidence_quality = classify_evidence_quality(
            text_value=row.text,
            claim_provenance=claim_provenance,
            intervention=intervention,
        )

        ranked.append(
            {
                "row": row,
                "similarity": hybrid_score,
                "vector_similarity": vector_similarity,
                "causal_relevance": causal_score,
                "matches": matches,
                "claim_provenance": claim_provenance,
                "evidence_quality": evidence_quality,
            }
        )

    # --------------------------------------------------------
    # STRICT CAUSAL FILTER
    # --------------------------------------------------------

    if intervention and metric:

        causal_matches = [
            item
            for item in ranked
            if (
                item["matches"]["intervention_matches"]
                and
                item["matches"]["metric_matches"]
            )
        ]

        if causal_matches:
            ranked = causal_matches

    # --------------------------------------------------------
    # Provenance-aware ranking
    # --------------------------------------------------------
    #
    # Direct causal-edge evidence is preferred over generic text
    # matches. Multi-intervention synthesis remains valid, but is
    # ranked after clean direct-edge evidence.
    provenance_priority = {
        "direct_edge": 2,
        "text_match": 1,
    }

    quality_priority = {
        "direct_edge": 2,
        "multi_intervention_synthesis": 1,
        "text_match": 0,
    }

    ranked.sort(
        key=lambda item: (
            provenance_priority.get(item["claim_provenance"], 0),
            quality_priority.get(item["evidence_quality"], 0),
            item["similarity"],
        ),
        reverse=True,
    )

    # --------------------------------------------------------
    # Deduplicate repeated evidence
    # --------------------------------------------------------

    deduplicated = []
    seen = set()

    for item in ranked:
        key = evidence_dedup_key(item["row"])

        if key in seen:
            continue

        seen.add(key)
        deduplicated.append(item)

    # --------------------------------------------------------
    # Return compatible result objects
    # --------------------------------------------------------

    output = []

    for item in deduplicated[:limit]:

        row = item["row"]

        output.append(
            EvidenceSearchResult(
                id=row.id,
                source_id=row.source_id,
                page=row.page,
                section=row.section,
                text=row.text,
                similarity=item["similarity"],
                vector_similarity=item["vector_similarity"],
                causal_relevance=item["causal_relevance"],
                claim_provenance=item["claim_provenance"],
                evidence_quality=item["evidence_quality"],
            )
        )

    return output


# ============================================================
# CAUSAL-AWARE SEARCH
# ============================================================

def search_causal_evidence(
    query: str,
    intervention: str,
    metric: str,
    source_id: str,
    relation: str | None = None,
    limit: int = 5,
):
    """Retrieve evidence for one specific causal relationship."""

    return search_evidence(
        query=query,
        limit=limit,
        source_id=source_id,
        intervention=intervention,
        metric=metric,
        relation=relation,
    )


# ============================================================
# SOURCE-SPECIFIC SEARCH
# ============================================================

def search_source_evidence(
    query: str,
    source_id: str,
    limit: int = 5,
    intervention: str | None = None,
    metric: str | None = None,
    relation: str | None = None,
):
    """
    Search only inside a specified scientific source.

    Causal metadata is forwarded to the hybrid retriever.
    """

    return search_evidence(
        query=query,
        limit=limit,
        source_id=source_id,
        intervention=intervention,
        metric=metric,
        relation=relation,
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    query = input(
        "Scientific question: "
    )

    results = search_evidence(
        query
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SCIENTIFIC EVIDENCE RESULTS"
    )

    print(
        "=" * 70
    )

    for i, row in enumerate(
        results,
        start=1,
    ):

        print(
            f"\n--- Result {i} ---"
        )

        print(
            f"Chunk: {row.id}"
        )

        print(
            f"Source: {row.source_id}"
        )

        print(
            f"Page: {row.page}"
        )

        print(
            f"Vector similarity: "
            f"{row.vector_similarity:.4f}"
        )

        print(
            f"Causal relevance: "
            f"{row.causal_relevance:.4f}"
        )

        print(
            f"Final similarity: "
            f"{row.similarity:.4f}"
        )

        print(
            f"Claim provenance: "
            f"{row.claim_provenance}"
        )

        print(
            f"Evidence quality: "
            f"{row.evidence_quality}"
        )

        print(
            f"\n{row.text}"
        )
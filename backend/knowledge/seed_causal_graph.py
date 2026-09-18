from sqlalchemy import text

from backend.db.database import engine


# ============================================================
# CURATED CAUSAL KNOWLEDGE
# ============================================================
#
# These are high-value relationships for the Darukaa.Earth
# reasoning layer.
#
# IMPORTANT:
# - Every edge must point to an existing canonical node.
# - Every claim has a source.
# - Quotes are kept as provenance.
# - These edges are provisional until scientifically reviewed.
#
# ============================================================


EDGES = [

    # ========================================================
    # COVER CROPS — SOIL CARBON
    # ========================================================

    {
        "src": "cover_crops",
        "dst": "soil_organic_carbon",
        "relation": "increases",
        "effect_type": "pct_change",
        "effect_size": 9.9,
        "source": "S02",
        "quote": (
            "Cover crops increased soil organic carbon "
            "in agroecosystems."
        ),
        "confidence": 0.90,
        "notes": "Synthesis evidence; magnitude depends on context.",
    },

    {
        "src": "cover_crops",
        "dst": "soil_organic_carbon",
        "relation": "increases",
        "effect_type": "absolute",
        "effect_size": 331,
        "effect_unit": "kg C ha-1 yr-1",
        "source": "S02",
        "quote": (
            "Cover crops increased soil organic carbon "
            "with measurable annual carbon inputs."
        ),
        "confidence": 0.90,
        "notes": "Reported synthesis estimate.",
    },

    # ========================================================
    # RESIDUE RETENTION — SOIL CARBON
    # ========================================================

    {
        "src": "residue_retention",
        "dst": "soil_organic_carbon",
        "relation": "increases",
        "effect_type": "pct_change",
        "effect_size": 9.9,
        "source": "S02",
        "quote": (
            "Crop residues contributed to increases in "
            "soil organic carbon."
        ),
        "confidence": 0.90,
        "notes": "Synthesis evidence.",
    },

    {
        "src": "residue_retention",
        "dst": "soil_organic_carbon",
        "relation": "increases",
        "effect_type": "absolute",
        "effect_size": 212,
        "effect_unit": "kg C ha-1 yr-1",
        "source": "S02",
        "quote": (
            "Crop residue retention increased soil organic "
            "carbon stocks."
        ),
        "confidence": 0.90,
        "notes": "Reported synthesis estimate.",
    },

    # ========================================================
    # MANURE — SOIL CARBON
    # ========================================================

    {
        "src": "manure_application",
        "dst": "soil_organic_carbon",
        "relation": "increases",
        "effect_type": "pct_change",
        "effect_size": 30.2,
        "source": "S02",
        "quote": (
            "Manure application increased soil organic carbon."
        ),
        "confidence": 0.92,
        "notes": "Synthesis evidence.",
    },

    # ========================================================
    # NITROGEN FERTILIZATION — SOIL CARBON
    # ========================================================

    {
        "src": "n_fertilization",
        "dst": "soil_organic_carbon",
        "relation": "increases",
        "effect_type": "qualitative",
        "source": "S09",
        "quote": (
            "Fertilizer effects on carbon accrual depend on "
            "initial soil organic carbon conditions."
        ),
        "confidence": 0.88,
        "notes": (
            "Context-dependent relationship; initial SOC "
            "is an important modifier."
        ),
    },

    # ========================================================
    # COVER CROPS — CROP YIELD
    # ========================================================

    {
        "src": "cover_crops",
        "dst": "crop_yield",
        "relation": "decreases",
        "effect_type": "qualitative",
        "source": "S15",
        "quote": (
            "Both legume and non-legume cover crops reduced "
            "cash crop yields."
        ),
        "confidence": 0.95,
        "notes": (
            "Dryland context; effect depends on cover-crop "
            "choice and water availability."
        ),
    },

    {
        "src": "cover_crops",
        "dst": "crop_yield",
        "relation": "increases",
        "effect_type": "pct_change",
        "effect_size": 6,
        "source": "S15",
        "quote": (
            "Cover crops increased cottonseed and lint yield "
            "by 6%."
        ),
        "confidence": 0.90,
        "notes": "Cotton-specific evidence.",
    },

    # ========================================================
    # COVER CROPS — WATER
    # ========================================================

    {
        "src": "cover_crops",
        "dst": "water_holding_capacity",
        "relation": "increases",
        "effect_type": "qualitative",
        "source": "S15",
        "quote": (
            "Cover crop management affects soil water "
            "availability for subsequent crops."
        ),
        "confidence": 0.80,
        "notes": (
            "Do not interpret this as direct evidence of "
            "increased soil water content in every context."
        ),
    },

    # ========================================================
    # CROP DIVERSIFICATION — BIODIVERSITY
    # ========================================================

    {
        "src": "crop_diversification",
        "dst": "biodiversity",
        "relation": "increases",
        "effect_type": "qualitative",
        "source": "S13",
        "quote": (
            "Crop diversification has positive effects on "
            "biodiversity."
        ),
        "confidence": 0.94,
        "notes": "Global synthesis/meta-analysis.",
    },

    {
        "src": "crop_diversification",
        "dst": "associated_biodiversity",
        "relation": "increases",
        "effect_type": "qualitative",
        "source": "S13",
        "quote": (
            "Crop diversification promotes biodiversity "
            "within agricultural systems."
        ),
        "confidence": 0.90,
        "notes": "Agricultural biodiversity context.",
    },

    # ========================================================
    # CROP DIVERSIFICATION — ECOSYSTEM SERVICES
    # ========================================================

    {
        "src": "crop_diversification",
        "dst": "pest_disease_control",
        "relation": "increases",
        "effect_type": "qualitative",
        "source": "S21",
        "quote": (
            "Agricultural diversification promotes multiple "
            "ecosystem services."
        ),
        "confidence": 0.86,
        "notes": "Relationship represents ecosystem-service benefits.",
    },

    {
        "src": "crop_diversification",
        "dst": "crop_yield",
        "relation": "increases",
        "effect_type": "qualitative",
        "source": "S21",
        "quote": (
            "Agricultural diversification promotes multiple "
            "ecosystem services without compromising yield."
        ),
        "confidence": 0.88,
        "notes": "Global synthesis; not a universal yield increase.",
    },

    # ========================================================
    # CROP ROTATION — SOIL CARBON
    # ========================================================

    {
        "src": "crop_rotation",
        "dst": "soil_organic_carbon",
        "relation": "increases",
        "effect_type": "qualitative",
        "source": "S02",
        "quote": (
            "Crop management practices influence soil organic "
            "carbon changes."
        ),
        "confidence": 0.75,
        "notes": "Broad synthesis relationship.",
    },

    # ========================================================
    # MINIMUM TILLAGE — SOIL CARBON
    # ========================================================

    {
        "src": "minimum_tillage",
        "dst": "soil_organic_carbon",
        "relation": "increases",
        "effect_type": "qualitative",
        "source": "S03",
        "quote": (
            "Conservation agriculture practices affect soil "
            "carbon sequestration."
        ),
        "confidence": 0.84,
        "notes": "Indian subcontinent meta-analysis.",
    },

    # ========================================================
    # MINIMUM TILLAGE — NITRATE LOSS
    # ========================================================

    {
        "src": "minimum_tillage",
        "dst": "water_quality",
        "relation": "increases",
        "effect_type": "qualitative",
        "source": "S20",
        "quote": (
            "No-tillage management affects nitrate loss from "
            "cropland systems."
        ),
        "confidence": 0.82,
        "notes": "Water-quality interpretation via nitrate loss.",
    },

    # ========================================================
    # AGROFORESTRY — BIODIVERSITY
    # ========================================================

    {
        "src": "agroforestry",
        "dst": "biodiversity",
        "relation": "increases",
        "effect_type": "qualitative",
        "source": "S25",
        "quote": (
            "Agroforestry enhances agroecosystem "
            "multifunctionality."
        ),
        "confidence": 0.84,
        "notes": "Global quantitative synthesis.",
    },

    # ========================================================
    # AGROFORESTRY — SOIL CARBON
    # ========================================================

    {
        "src": "agroforestry",
        "dst": "soil_organic_carbon",
        "relation": "increases",
        "effect_type": "qualitative",
        "source": "S25",
        "quote": (
            "Agroforestry affects multiple ecosystem "
            "functions including soil-related functions."
        ),
        "confidence": 0.80,
        "notes": "Global synthesis; context dependent.",
    },

    # ========================================================
    # AGROFORESTRY — SOIL QUALITY
    # ========================================================

    {
        "src": "agroforestry",
        "dst": "soil_quality",
        "relation": "increases",
        "effect_type": "qualitative",
        "source": "S25",
        "quote": (
            "Agroforestry enhances agroecosystem "
            "multifunctionality."
        ),
        "confidence": 0.78,
        "notes": "Broad multifunctionality evidence.",
    },

    # ========================================================
    # INTERCROPPING — BIODIVERSITY
    # ========================================================

    {
        "src": "intercropping",
        "dst": "biodiversity",
        "relation": "increases",
        "effect_type": "qualitative",
        "source": "S13",
        "quote": (
            "Crop diversification has positive effects "
            "on biodiversity."
        ),
        "confidence": 0.82,
        "notes": "Intercropping treated as a diversification practice.",
    },

    # ========================================================
    # INTERCROPPING — PEST CONTROL
    # ========================================================

    {
        "src": "intercropping",
        "dst": "pest_disease_control",
        "relation": "increases",
        "effect_type": "qualitative",
        "source": "S21",
        "quote": (
            "Agricultural diversification promotes multiple "
            "ecosystem services."
        ),
        "confidence": 0.78,
        "notes": "Diversification mechanism; context dependent.",
    },

    # ========================================================
    # VARIETY MIXTURES — BIODIVERSITY
    # ========================================================

    {
        "src": "variety_mixtures",
        "dst": "associated_biodiversity",
        "relation": "increases",
        "effect_type": "qualitative",
        "source": "S13",
        "quote": (
            "Crop diversification promotes biodiversity "
            "in agricultural systems."
        ),
        "confidence": 0.78,
        "notes": "Diversification category.",
    },

    # ========================================================
    # CONSERVATION AGRICULTURE — SOIL QUALITY
    # ========================================================

    {
        "src": "conservation_agriculture",
        "dst": "soil_quality",
        "relation": "increases",
        "effect_type": "qualitative",
        "source": "S10",
        "quote": (
            "Conservation agriculture is based on principles "
            "including minimum soil disturbance, permanent "
            "soil cover, and crop diversification."
        ),
        "confidence": 0.88,
        "notes": "FAO principles; umbrella intervention.",
    },

    # ========================================================
    # SOIL COVER — SOIL EROSION
    # ========================================================

    {
        "src": "soil_cover",
        "dst": "soil_erosion",
        "relation": "decreases",
        "effect_type": "qualitative",
        "source": "S10",
        "quote": (
            "Permanent soil cover is a core conservation "
            "agriculture principle."
        ),
        "confidence": 0.82,
        "notes": "Mechanistic/principle-level relationship.",
    },

    # ========================================================
    # RESIDUE RETENTION — SOIL EROSION
    # ========================================================

    {
        "src": "residue_retention",
        "dst": "soil_erosion",
        "relation": "decreases",
        "effect_type": "qualitative",
        "source": "S10",
        "quote": (
            "Permanent soil cover is a core principle of "
            "conservation agriculture."
        ),
        "confidence": 0.76,
        "notes": "Residue retention contributes to soil cover.",
    },
]


# ============================================================
# VALIDATION
# ============================================================

def validate_edge(edge: dict, nodes: set[str], sources: set[str]):

    required = {
        "src",
        "dst",
        "relation",
        "effect_type",
        "source",
        "quote",
        "confidence",
    }

    missing = required - edge.keys()

    if missing:
        raise ValueError(
            f"Missing fields: {missing}"
        )

    if edge["src"] not in nodes:
        raise ValueError(
            f"Unknown source node: {edge['src']}"
        )

    if edge["dst"] not in nodes:
        raise ValueError(
            f"Unknown destination node: {edge['dst']}"
        )

    if edge["source"] not in sources:
        raise ValueError(
            f"Unknown scientific source: {edge['source']}"
        )

    if not 0 <= edge["confidence"] <= 1:
        raise ValueError(
            f"Invalid confidence: {edge['confidence']}"
        )


# ============================================================
# DUPLICATE CHECK
# ============================================================

def edge_exists(edge: dict) -> bool:

    sql = text("""
        SELECT EXISTS (
            SELECT 1
            FROM edge
            WHERE source_id = :source_id
              AND src_node = :src_node
              AND dst_node = :dst_node
              AND relation = CAST(:relation AS relation_kind)
              AND effect_type = CAST(:effect_type AS effect_type)
              AND effect_size IS NOT DISTINCT FROM :effect_size
              AND effect_unit IS NOT DISTINCT FROM :effect_unit
              AND quote = :quote
        )
    """)

    with engine.connect() as connection:

        return bool(
            connection.execute(
                sql,
                {
                    "source_id": edge["source"],
                    "src_node": edge["src"],
                    "dst_node": edge["dst"],
                    "relation": edge["relation"],
                    "effect_type": edge["effect_type"],
                    "effect_size": edge.get("effect_size"),
                    "effect_unit": edge.get("effect_unit"),
                    "quote": edge["quote"],
                },
            ).scalar()
        )


# ============================================================
# INSERT
# ============================================================

def insert_edge(edge: dict):

    sql = text("""
        INSERT INTO edge (
            src_node,
            dst_node,
            relation,
            effect_type,
            effect_size,
            effect_unit,
            ci_low,
            ci_high,
            ci_level,
            lag_years,
            time_to_full_effect,
            evidence_tier,
            source_id,
            chunk_id,
            quote,
            confidence,
            verification_status,
            extraction_method,
            notes
        )
        VALUES (
            :src_node,
            :dst_node,
            CAST(:relation AS relation_kind),
            CAST(:effect_type AS effect_type),
            :effect_size,
            :effect_unit,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            (
                SELECT evidence_tier
                FROM source
                WHERE id = :source_id
            ),
            :source_id,
            NULL,
            :quote,
            :confidence,
            CAST('provisional' AS verification_status),
            'curated_seed',
            :notes
        )
        RETURNING id
    """)

    with engine.begin() as connection:

        return connection.execute(
            sql,
            {
                "src_node": edge["src"],
                "dst_node": edge["dst"],
                "relation": edge["relation"],
                "effect_type": edge["effect_type"],
                "effect_size": edge.get("effect_size"),
                "effect_unit": edge.get("effect_unit"),
                "source_id": edge["source"],
                "quote": edge["quote"],
                "confidence": edge["confidence"],
                "notes": edge.get("notes"),
            },
        ).scalar_one()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DARUKAA.EARTH — CURATED CAUSAL GRAPH SEED")
    print("=" * 70)

    with engine.connect() as connection:

        nodes = {
            row[0]
            for row in connection.execute(
                text("SELECT id FROM node")
            ).fetchall()
        }

        sources = {
            row[0]
            for row in connection.execute(
                text("SELECT id FROM source")
            ).fetchall()
        }

    print(f"Nodes available : {len(nodes)}")
    print(f"Sources available: {len(sources)}")
    print(f"Edges to process : {len(EDGES)}")
    print()

    inserted = 0
    skipped = 0

    for i, edge in enumerate(EDGES, start=1):

        print(
            f"[{i}/{len(EDGES)}] "
            f"{edge['src']} "
            f"--{edge['relation']}--> "
            f"{edge['dst']}"
        )

        validate_edge(
            edge,
            nodes,
            sources,
        )

        if edge_exists(edge):

            print("  ↳ already exists; skipping")
            skipped += 1
            continue

        edge_id = insert_edge(edge)

        print(
            f"  ↳ INSERTED edge #{edge_id}"
        )

        inserted += 1

    print()
    print("=" * 70)
    print("SEED COMPLETE")
    print("=" * 70)
    print(f"Inserted : {inserted}")
    print(f"Skipped  : {skipped}")
    print("=" * 70)


if __name__ == "__main__":
    main()
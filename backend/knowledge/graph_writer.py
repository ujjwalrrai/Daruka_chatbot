from sqlalchemy import text

from backend.db.database import engine
from backend.knowledge.claim_schema import ExtractedClaim


def edge_exists(
    source_id: str,
    chunk_id: int,
    claim: ExtractedClaim,
) -> bool:

    sql = text("""
        SELECT EXISTS (
            SELECT 1
            FROM edge
            WHERE source_id = :source_id
              AND chunk_id = :chunk_id
              AND src_node = :src_node
              AND dst_node = :dst_node
              AND relation = CAST(:relation AS relation_kind)
        )
    """)

    with engine.connect() as connection:
        return connection.execute(
            sql,
            {
                "source_id": source_id,
                "chunk_id": chunk_id,
                "src_node": claim.src_node,
                "dst_node": claim.dst_node,
                "relation": claim.relation,
            },
        ).scalar()


def write_claim(
    source_id: str,
    chunk_id: int,
    claim: ExtractedClaim,
) -> int | None:

    if edge_exists(
        source_id=source_id,
        chunk_id=chunk_id,
        claim=claim,
    ):
        print(
            f"SKIP duplicate: "
            f"{claim.src_node} -> {claim.dst_node}"
        )
        return None

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
            :ci_low,
            :ci_high,
            :ci_level,
            :lag_years,
            :time_to_full_effect,
            (
                SELECT evidence_tier
                FROM source
                WHERE id = :source_id
            ),
            :source_id,
            :chunk_id,
            :quote,
            :confidence,
            CAST('provisional' AS verification_status),
            'langchain_llm',
            :notes
        )
        RETURNING id
    """)

    with engine.begin() as connection:
        edge_id = connection.execute(
            sql,
            {
                "src_node": claim.src_node,
                "dst_node": claim.dst_node,
                "relation": claim.relation,
                "effect_type": claim.effect_type,
                "effect_size": claim.effect_size,
                "effect_unit": claim.effect_unit,
                "ci_low": claim.ci_low,
                "ci_high": claim.ci_high,
                "ci_level": claim.ci_level,
                "lag_years": claim.lag_years,
                "time_to_full_effect": (
                    claim.time_to_full_effect
                ),
                "source_id": source_id,
                "chunk_id": chunk_id,
                "quote": claim.quote,
                "confidence": claim.confidence,
                "notes": claim.notes,
            },
        ).scalar_one()

    print(
        f"INSERTED edge #{edge_id}: "
        f"{claim.src_node} "
        f"--{claim.relation}--> "
        f"{claim.dst_node}"
    )

    return edge_id
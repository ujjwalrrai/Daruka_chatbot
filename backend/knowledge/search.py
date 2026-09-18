from __future__ import annotations

from sqlalchemy import text

from backend.db.database import engine
from backend.knowledge.embeddings import embed_texts


# ============================================================
# EVIDENCE SEARCH
# ============================================================

def search_evidence(
    query: str,
    limit: int = 5,
    source_id: str | None = None,
):
    """
    Semantic evidence retrieval using pgvector.

    Optional source_id allows the reasoning engine to restrict
    retrieval to the source attached to a causal edge.
    """

    embedding = embed_texts(
        [query]
    )[0]

    source_filter = ""

    params = {
        "embedding": str(embedding),
        "limit": limit,
    }

    if source_id:

        source_filter = """
            AND ec.source_id = :source_id
        """

        params["source_id"] = source_id

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
                <=> CAST(:embedding AS vector)
            ) AS similarity

        FROM evidence_chunk ec

        WHERE
            ec.embedding IS NOT NULL

            {source_filter}

            -- Ignore extremely short fragments.
            AND LENGTH(
                TRIM(ec.text)
            ) >= 250

            -- Avoid obvious title-only chunks.
            AND ec.text NOT LIKE 'Productivity limits%'

            AND ec.text NOT LIKE
                'Cover crop impacts%'

            AND ec.text NOT LIKE
                'Animal manure application%'

            AND ec.text NOT LIKE
                'Roots contribute%'

        ORDER BY
            ec.embedding
            <=> CAST(:embedding AS vector)

        LIMIT :limit
        """
    )

    with engine.connect() as connection:

        rows = connection.execute(
            sql,
            params,
        ).fetchall()

    return rows


# ============================================================
# SOURCE-SPECIFIC SEARCH
# ============================================================

def search_source_evidence(
    query: str,
    source_id: str,
    limit: int = 5,
):
    """
    Search evidence only inside a specific scientific source.
    """

    return search_evidence(
        query=query,
        limit=limit,
        source_id=source_id,
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
            f"Similarity: "
            f"{row.similarity:.4f}"
        )

        print(
            f"\n{row.text}"
        )
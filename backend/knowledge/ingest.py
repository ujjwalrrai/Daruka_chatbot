from pathlib import Path

from sqlalchemy import text

from backend.db.database import engine
from backend.knowledge.loader import load_source_document, load_source_pdf
from backend.knowledge.chunker import chunk_pages
from backend.knowledge.embeddings import embed_texts
from backend.knowledge.sources import SOURCE_REGISTRY


DATA_DIR = Path("data/sources")


def register_source(source_id: str, metadata: dict) -> Path:
    """
    Insert or update source metadata in the source table.

    Returns the expected local PDF path.
    """

    local_path = DATA_DIR / f"{source_id}.pdf"

    query = text(
        """
        INSERT INTO source (
            id,
            title,
            authors,
            year,
            publisher,
            doi,
            url,
            study_design,
            evidence_tier,
            geo_scope,
            biome,
            n_studies,
            credibility,
            local_path,
            notes
        )
        VALUES (
            :id,
            :title,
            :authors,
            :year,
            :publisher,
            :doi,
            :url,
            :study_design,
            :evidence_tier,
            :geo_scope,
            :biome,
            :n_studies,
            :credibility,
            :local_path,
            :notes
        )
        ON CONFLICT (id)
        DO UPDATE SET
            title = EXCLUDED.title,
            authors = EXCLUDED.authors,
            year = EXCLUDED.year,
            publisher = EXCLUDED.publisher,
            doi = EXCLUDED.doi,
            url = EXCLUDED.url,
            study_design = EXCLUDED.study_design,
            evidence_tier = EXCLUDED.evidence_tier,
            geo_scope = EXCLUDED.geo_scope,
            biome = EXCLUDED.biome,
            n_studies = EXCLUDED.n_studies,
            credibility = EXCLUDED.credibility,
            local_path = EXCLUDED.local_path,
            notes = EXCLUDED.notes
        """
    )

    values = {
        "id": source_id,
        "title": metadata["title"],
        "authors": metadata["authors"],
        "year": metadata["year"],
        "publisher": metadata["publisher"],
        "doi": metadata["doi"],
        "url": metadata["url"],
        "study_design": metadata["study_design"],
        "evidence_tier": metadata["evidence_tier"],
        "geo_scope": metadata["geo_scope"],
        "biome": metadata["biome"],
        "n_studies": metadata["n_studies"],
        "credibility": metadata["credibility"],
        "local_path": str(local_path),
        "notes": metadata["notes"],
    }

    with engine.begin() as connection:
        connection.execute(query, values)

    print(f"Source {source_id} registered")

    return local_path


def source_already_ingested(source_id: str) -> bool:
    """
    Check whether evidence chunks already exist for this source.
    """

    query = text(
        """
        SELECT COUNT(*)
        FROM evidence_chunk
        WHERE source_id = :source_id
        """
    )

    with engine.connect() as connection:
        count = connection.execute(
            query,
            {"source_id": source_id},
        ).scalar()

    return count > 0


def insert_chunks(
    source_id: str,
    chunks: list[dict],
    embeddings: list[list[float]],
):
    """
    Insert evidence chunks and their embeddings.
    """

    query = text(
        """
        INSERT INTO evidence_chunk (
            source_id,
            page,
            section,
            text,
            embedding,
            token_count
        )
        VALUES (
            :source_id,
            :page,
            :section,
            :text,
            CAST(:embedding AS vector),
            :token_count
        )
        """
    )

    rows = []

    for chunk, embedding in zip(chunks, embeddings):
        rows.append(
            {
                "source_id": source_id,
                "page": chunk["page"],
                "section": chunk.get("section"),
                "text": chunk["text"],
                "embedding": str(embedding),
                "token_count": len(
                    chunk["text"].split()
                ),
            }
        )

    if not rows:
        print("No chunks to insert.")
        return

    with engine.begin() as connection:
        connection.execute(query, rows)

    print(f"Inserted {len(rows)} evidence chunks")


def ingest_source(source_id: str):
    """
    Ingest one scientific source.

    Steps:
        1. Register source metadata
        2. Check duplicate protection
        3. Load PDF
        4. Extract pages
        5. Chunk text
        6. Generate embeddings
        7. Store evidence chunks
    """

    if source_id not in SOURCE_REGISTRY:
        raise ValueError(
            f"Unknown source ID: {source_id}"
        )

    metadata = SOURCE_REGISTRY[source_id]

    print()
    print("=" * 60)
    print(f"INGESTING {source_id}")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. REGISTER SOURCE
    # ---------------------------------------------------------

    local_path = register_source(
        source_id,
        metadata,
    )

    # ---------------------------------------------------------
    # 2. DUPLICATE PROTECTION
    # ---------------------------------------------------------

    if source_already_ingested(source_id):
        print(
            f"{source_id} already has evidence chunks."
        )
        print(
            "Skipping to prevent duplicate evidence."
        )
        return

    # ---------------------------------------------------------
    # 3. LOAD SOURCE PDF
    # ---------------------------------------------------------

    try:
        pages = load_source_document(
            source_id=source_id,
            metadata=metadata,
            local_path=str(local_path),
        )

    except ValueError as error:
        print(f"Skipping {source_id}: {error}")
        return

    # ---------------------------------------------------------
    # 4. CHUNK TEXT
    # ---------------------------------------------------------

    chunks = chunk_pages(pages)

    if not chunks:
        print(
            f"No text chunks were created for {source_id}."
        )
        return

    # ---------------------------------------------------------
    # 5. GENERATE EMBEDDINGS
    # ---------------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embed_texts(texts)

    if not embeddings:
        print(
            f"No embeddings generated for {source_id}."
        )
        return

    print(
        f"Embedding dimension: "
        f"{len(embeddings[0])}"
    )

    # ---------------------------------------------------------
    # 6. INSERT INTO DATABASE
    # ---------------------------------------------------------

    insert_chunks(
        source_id,
        chunks,
        embeddings,
    )


def main():
    print("=" * 60)
    print(
        "DARUKAA.EARTH — SCIENTIFIC KNOWLEDGE INGESTION"
    )
    print("=" * 60)

    for source_id in SOURCE_REGISTRY:

        try:
            ingest_source(source_id)

        except Exception as error:
            print()
            print(
                f"ERROR while ingesting {source_id}:"
            )
            print(error)

            # Continue with other sources rather than
            # terminating the entire ingestion run.
            continue

    print()
    print("=" * 60)
    print("INGESTION RUN COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

from concurrent.futures import ThreadPoolExecutor, as_completed

from sqlalchemy import text

from backend.db.database import engine
from backend.knowledge.claim_extractor import extract_claims
from backend.knowledge.graph_writer import write_claim


# ============================================================
# CONFIG
# ============================================================

SOURCE_IDS = (
    "S15",
    "S20",
    "S21",
    "S22",
    "S23",
    "S24",
    "S25",
)

# Keep this low for Groq stability.
MAX_WORKERS = 2

# For now, only process this many chunks.
# Set to None after the test succeeds.
TEST_LIMIT = 20


# ============================================================
# DATABASE
# ============================================================

def get_nodes():
    sql = text("""
        SELECT
            id,
            kind,
            canonical_name,
            description,
            aliases
        FROM node
        ORDER BY id
    """)

    with engine.connect() as connection:
        rows = connection.execute(sql).mappings().all()

    return [dict(row) for row in rows]


def get_chunks():
    sql = text("""
        SELECT
            id,
            source_id,
            page,
            section,
            text
        FROM evidence_chunk
        WHERE source_id = ANY(:source_ids)
        ORDER BY source_id, id
    """)

    with engine.connect() as connection:
        rows = connection.execute(
            sql,
            {
                "source_ids": list(SOURCE_IDS)
            }
        ).mappings().all()

    return [dict(row) for row in rows]


def get_status(chunk_id):
    sql = text("""
        SELECT status
        FROM evidence_extraction
        WHERE chunk_id = :chunk_id
    """)

    with engine.connect() as connection:
        row = connection.execute(
            sql,
            {
                "chunk_id": chunk_id
            }
        ).fetchone()

    return row[0] if row else None


def create_checkpoint(chunk_id, source_id):
    sql = text("""
        INSERT INTO evidence_extraction (
            chunk_id,
            source_id,
            status
        )
        VALUES (
            :chunk_id,
            :source_id,
            'processing'
        )
        ON CONFLICT (chunk_id) DO NOTHING
    """)

    with engine.begin() as connection:
        connection.execute(
            sql,
            {
                "chunk_id": chunk_id,
                "source_id": source_id
            }
        )


def increment_attempt(chunk_id):
    sql = text("""
        UPDATE evidence_extraction
        SET
            attempts = attempts + 1,
            status = 'processing'
        WHERE chunk_id = :chunk_id
    """)

    with engine.begin() as connection:
        connection.execute(
            sql,
            {
                "chunk_id": chunk_id
            }
        )


def mark_completed(chunk_id, claims_found):
    sql = text("""
        UPDATE evidence_extraction
        SET
            status = 'completed',
            claims_found = :claims_found,
            error = NULL,
            processed_at = NOW()
        WHERE chunk_id = :chunk_id
    """)

    with engine.begin() as connection:
        connection.execute(
            sql,
            {
                "chunk_id": chunk_id,
                "claims_found": claims_found
            }
        )


def mark_failed(chunk_id, error):
    sql = text("""
        UPDATE evidence_extraction
        SET
            status = 'failed',
            error = :error
        WHERE chunk_id = :chunk_id
    """)

    with engine.begin() as connection:
        connection.execute(
            sql,
            {
                "chunk_id": chunk_id,
                "error": str(error)[:2000]
            }
        )


# ============================================================
# SINGLE CHUNK
# ============================================================

def process_chunk(chunk, nodes):

    chunk_id = chunk["id"]
    source_id = chunk["source_id"]

    try:

        status = get_status(chunk_id)

        # Completed chunks are never processed again.
        if status == "completed":
            return {
                "status": "skipped",
                "chunk_id": chunk_id,
                "claims": 0,
                "edges": 0,
            }

        create_checkpoint(
            chunk_id,
            source_id
        )

        increment_attempt(
            chunk_id
        )

        print(
            f"\n[LLM] chunk={chunk_id} "
            f"source={source_id}"
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # No local keyword filter.
        # Let the scientific extractor decide.
        # ----------------------------------------------------

        claims = extract_claims(
            evidence_text=chunk["text"],
            source_id=source_id,
            available_nodes=nodes,
        )

        # ----------------------------------------------------
        # Write claims
        # ----------------------------------------------------

        edges = 0

        for claim in claims:

            edge_id = write_claim(
                source_id=source_id,
                chunk_id=chunk_id,
                claim=claim,
            )

            if edge_id is not None:
                edges += 1

        mark_completed(
            chunk_id,
            len(claims)
        )

        return {
            "status": "completed",
            "chunk_id": chunk_id,
            "claims": len(claims),
            "edges": edges,
        }

    except Exception as exc:

        mark_failed(
            chunk_id,
            repr(exc)
        )

        return {
            "status": "failed",
            "chunk_id": chunk_id,
            "claims": 0,
            "edges": 0,
            "error": repr(exc),
        }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("DARUKAA.EARTH CEE")
    print("SAFE CONCURRENT CLAIM EXTRACTION")
    print("=" * 80)

    nodes = get_nodes()
    chunks = get_chunks()

    print()
    print(f"Evidence chunks : {len(chunks)}")
    print(f"Canonical nodes : {len(nodes)}")
    print(f"Workers         : {MAX_WORKERS}")
    print(f"Test limit      : {TEST_LIMIT}")
    print()

    # --------------------------------------------------------
    # Only pending chunks
    # --------------------------------------------------------

    pending = []

    for chunk in chunks:

        status = get_status(
            chunk["id"]
        )

        if status != "completed":
            pending.append(chunk)

    print(
        f"Pending chunks  : {len(pending)}"
    )

    # --------------------------------------------------------
    # TEST LIMIT
    # --------------------------------------------------------

    if TEST_LIMIT is not None:
        pending = pending[:TEST_LIMIT]

    print(
        f"Will process    : {len(pending)}"
    )

    print()

    if not pending:
        print("Nothing to process.")
        return

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    processed = 0
    failed = 0
    skipped = 0
    total_claims = 0
    total_edges = 0

    total = len(pending)

    # --------------------------------------------------------
    # Thread pool
    # --------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = [
            executor.submit(
                process_chunk,
                chunk,
                nodes
            )
            for chunk in pending
        ]

        for future in as_completed(futures):

            result = future.result()

            processed += 1

            if result["status"] == "failed":
                failed += 1

            elif result["status"] == "skipped":
                skipped += 1

            total_claims += result["claims"]
            total_edges += result["edges"]

            print(
                f"\n"
                f"[{processed}/{total}] "
                f"chunk={result['chunk_id']} "
                f"status={result['status']} "
                f"claims={result['claims']} "
                f"edges={result['edges']}"
            )

            if result.get("error"):
                print(
                    f"ERROR: {result['error']}"
                )

            print(
                f"TOTAL "
                f"claims={total_claims} "
                f"edges={total_edges} "
                f"failed={failed}"
            )

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    print(
        f"Processed : {processed}"
    )

    print(
        f"Skipped   : {skipped}"
    )

    print(
        f"Failed    : {failed}"
    )

    print(
        f"Claims    : {total_claims}"
    )

    print(
        f"Edges     : {total_edges}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
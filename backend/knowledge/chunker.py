from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1800,
    chunk_overlap=250,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
)


def chunk_pages(
    pages: list[dict],
) -> list[dict]:

    chunks = []

    for page in pages:

        text = page.get(
            "text",
            "",
        ).strip()

        if not text:
            continue

        page_chunks = splitter.split_text(
            text
        )

        for chunk in page_chunks:

            chunk = chunk.strip()

            if not chunk:
                continue

            chunks.append({
                "page": page.get("page"),
                "section": page.get("section"),
                "text": chunk,
            })

    print(
        f"Created {len(chunks)} chunks"
    )

    return chunks
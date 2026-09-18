from pathlib import Path

import requests
import pymupdf


HEADERS = {
    "User-Agent": (
        "DarukaaEarthCEE/1.0 "
        "(scientific-evidence-ingestion)"
    )
}


def download_pdf(
    url: str,
    output_path: str,
) -> Path:

    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Downloading PDF: {url}")

    response = requests.get(
        url,
        timeout=60,
        headers=HEADERS,
        allow_redirects=True,
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "content-type",
        "",
    ).lower()

    if (
        "pdf" not in content_type
        and not response.content.startswith(b"%PDF")
    ):
        raise ValueError(
            "URL did not return a PDF. "
            f"Content-Type={content_type}, "
            f"Final URL={response.url}"
        )

    path.write_bytes(response.content)

    print(f"Saved: {path}")
    print(
        f"Size: "
        f"{path.stat().st_size / 1024 / 1024:.2f} MB"
    )

    return path


def load_pdf(
    pdf_path: str,
) -> list[dict]:

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(
        document,
        start=1,
    ):

        text = page.get_text("text").strip()

        if not text:
            continue

        pages.append({
            "page": page_number,
            "section": None,
            "text": text,
        })

    document.close()

    print(
        f"Extracted {len(pages)} pages"
    )

    return pages


def load_pmc_bioc(
    pmcid: str,
) -> list[dict]:

    url = (
        "https://www.ncbi.nlm.nih.gov/"
        "research/bionlp/RESTful/pmcoa.cgi/"
        f"BioC_json/{pmcid}/unicode"
    )

    print(
        f"Downloading PMC full text: {pmcid}"
    )

    print(
        f"BioC endpoint: {url}"
    )

    response = requests.get(
        url,
        timeout=60,
        headers=HEADERS,
    )

    response.raise_for_status()

    try:
        data = response.json()

    except ValueError as error:

        raise ValueError(
            f"PMC BioC response was not valid JSON "
            f"for {pmcid}"
        ) from error

    passages = []

    # ---------------------------------------------------------
    # BioC structure:
    #
    # [
    #   {
    #       "documents": [
    #           {
    #               "passages": [...]
    #           }
    #       ]
    #   }
    # ]
    # ---------------------------------------------------------

    for collection in data:

        for document in collection.get(
            "documents",
            [],
        ):

            for passage in document.get(
                "passages",
                [],
            ):

                text = passage.get(
                    "text",
                    "",
                ).strip()

                if not text:
                    continue

                infons = passage.get(
                    "infons",
                    {},
                )

                section = (
                    infons.get("section")
                    or infons.get("type")
                    or None
                )

                passages.append({
                    "page": None,
                    "section": section,
                    "text": text,
                })

    if not passages:

        raise ValueError(
            f"No full-text passages returned "
            f"for {pmcid}"
        )

    print(
        f"Extracted {len(passages)} "
        f"scientific passages from PMC"
    )

    return passages


def load_source_document(
    source_id: str,
    metadata: dict,
    local_path: str,
) -> list[dict]:

    path = Path(local_path)

    # =========================================================
    # 1. EXISTING LOCAL PDF
    # =========================================================

    if path.exists():

        print(
            f"Using existing local document: "
            f"{path}"
        )

        return load_pdf(
            str(path)
        )

    # =========================================================
    # 2. PMC OPEN-ACCESS FULL TEXT
    # =========================================================

    pmcid = metadata.get("pmcid")

    if pmcid:

        return load_pmc_bioc(
            pmcid
        )

    # =========================================================
    # 3. DIRECT PDF URL
    # =========================================================

    pdf_url = metadata.get("pdf_url")

    if pdf_url:

        download_pdf(
            pdf_url,
            str(path),
        )

        return load_pdf(
            str(path)
        )

    # =========================================================
    # 4. NOTHING AVAILABLE
    # =========================================================

    raise ValueError(
        f"{source_id} has no usable full-text source. "
        "Provide a PMCID or verified PDF URL."
    )


# =============================================================
# BACKWARD COMPATIBILITY
# =============================================================

def load_source_pdf(
    source_id: str,
    pdf_url: str | None,
    local_path: str,
) -> list[dict]:

    path = Path(local_path)

    if not path.exists():

        if not pdf_url:

            raise ValueError(
                f"{source_id} has no verified PDF URL. "
                "Provide the PDF manually or add a "
                "verified open-access PDF URL."
            )

        download_pdf(
            pdf_url,
            local_path,
        )

    return load_pdf(
        local_path
    )
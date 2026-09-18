import requests
from bs4 import BeautifulSoup


def load_web_page(url: str) -> str:
    """
    Download a web page and extract readable text.
    """

    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "Chrome/153.0 Safari/537.36"
            )
        },
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove things that are not useful evidence.
    for element in soup(
        ["script", "style", "noscript", "nav", "footer", "header"]
    ):
        element.decompose()

    text = soup.get_text("\n")

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return "\n".join(lines)
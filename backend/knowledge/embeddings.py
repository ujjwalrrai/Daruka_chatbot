from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model = None


def get_model():

    global _model

    if _model is None:

        print(
            f"Loading embedding model: "
            f"{MODEL_NAME}"
        )

        _model = SentenceTransformer(
            MODEL_NAME
        )

        print(
            "Embedding model loaded"
        )

    return _model


def embed_texts(
    texts: list[str],
) -> list[list[float]]:

    model = get_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings.tolist()
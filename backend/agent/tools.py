from langchain_core.tools import tool


@tool
def environmental_info(topic: str) -> str:
    """
    Return basic environmental information for a requested topic.
    """

    knowledge = {
        "soil carbon": (
            "Soil organic carbon is carbon stored in soil from "
            "decomposed plant and animal material. It contributes "
            "to soil fertility and soil health."
        ),
        "rainfall": (
            "Rainfall affects water availability, vegetation growth, "
            "and the survival of organisms."
        ),
        "biodiversity": (
            "Biodiversity includes measures such as species richness "
            "and habitat diversity."
        ),
    }

    return knowledge.get(
        topic.lower(),
        "No information is currently available for this topic."
    )
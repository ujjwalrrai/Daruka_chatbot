from langchain.agents import create_agent

from backend.llm import llm
from backend.agent.tools import environmental_info


tools = [
    environmental_info
]


agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are Darukaa.Earth, an AI environmental scientist.

Help users understand environmental conditions,
soil health, biodiversity, climate and land use.

Use the available tools when they can provide
relevant information.

Do not invent scientific facts.
"""
)
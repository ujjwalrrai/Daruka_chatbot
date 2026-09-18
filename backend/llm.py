import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set")

llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)
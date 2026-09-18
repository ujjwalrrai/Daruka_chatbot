import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0,
    google_api_key=api_key,
)
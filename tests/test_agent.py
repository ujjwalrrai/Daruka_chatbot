from backend.agent.agent import agent


response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Tell me about soil carbon."
            }
        ]
    }
)


print("\n--- AGENT RESPONSE ---\n")

print(response["messages"][-1].content)
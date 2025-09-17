import asyncio
from langchain_ollama import ChatOllama
from mcp_use import MCPAgent, MCPClient
import os

async def main():

    # Create MCPClient from configuration dictionary
    # client = MCPClient.from_config_file(os.path.join(os.path.dirname(__file__), "mcp.json"))
    client = MCPClient.from_config_file(os.path.join(os.path.dirname(__file__), "mcp-http.json"))

    # Create LLM
    llm = ChatOllama(base_url="http://localhost:11434", model="gpt-oss:20b")

    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, max_steps=30, use_server_manager=True)

    # Run the query
    result = await agent.run(
        "What's the weather in New York and the 3-day forecast? Also, what's 12 multiplied by 342?",
    )
    print(f"\nResult: {result}")

if __name__ == "__main__":
    asyncio.run(main())
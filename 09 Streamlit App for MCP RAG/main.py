from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate

# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

server_params = StdioServerParameters(
    command="uv",
    # Make sure to update to the full absolute path to your server.py file
    args=["--directory",
            "D:\\Courses\\Udemy\\MCP Mastery - Claude and LangChain\\08 MCP RAG with LangChain",
            "run",
            "server.py"])



import asyncio

model = ChatOllama(model="qwen3", base_url="http://localhost:11434/")

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            print(f"Loaded tools: {[tool.name for tool in tools]}")

            # Create and run the agent
            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke({"messages": "show me db information?"})

            response = agent_response.get('messages')[-1].content
            print("Final response:", response)

    print(f"Agent response: {agent_response}")
    print("type of agent response:", type(agent_response))

if __name__ == "__main__":
    asyncio.run(main())
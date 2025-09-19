# client.py - Research Assistant with Firecrawl and LangGraph
import asyncio
from typing import List, Annotated
from typing_extensions import TypedDict

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient

from dotenv import load_dotenv
import os
import json

load_dotenv()

# Configuration
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

# Multi-server MCP client configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
mcp_config_path = os.path.join(current_dir, "mcp.json")
mcp_json = json.load(open(mcp_config_path, 'r'))
mcp_json["firecrawl_server"]["env"] = {
    "FIRECRAWL_API_KEY": FIRECRAWL_API_KEY
}


client = MultiServerMCPClient(mcp_json)

async def create_research_agent():
    """Create a LangGraph agent with research and web crawling capabilities."""
    
    # Initialize LLM
    llm = ChatOllama(model="qwen3", base_url="http://localhost:11434/")
    
    # Get tools from all MCP servers
    tools = await client.get_tools()
    llm_with_tools = llm.bind_tools(tools)
    
    # System prompt for research assistant
    system_message = """You are an advanced research assistant with access to web crawling and knowledge storage capabilities.

                        Your abilities:
                        1. **Web Research**: Use Firecrawl tools to scrape and analyze web content
                        2. **Knowledge Storage**: Save important research findings to vector databases organized by topic
                        3. **Information Retrieval**: Search through previously saved research using semantic similarity
                        4. **Research Management**: Organize and manage research topics

                        When conducting research:
                        - Always save important findings to the appropriate topic database
                        - Search existing knowledge first before crawling new content
                        - Provide comprehensive, well-structured responses
                        - Cite sources when possible

                        Available commands:
                        - Regular conversation for research questions
                        - The system will automatically use the best tools for your requests"""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder("messages")
    ])
    
    chat_llm = prompt_template | llm_with_tools
    
    # Define state
    class State(TypedDict):
        messages: Annotated[List[AnyMessage], add_messages]
    
    # Chat node
    def chat_node(state: State) -> State:
        response = chat_llm.invoke({"messages": state["messages"]})
        return {"messages": [response]}
    
    # Build graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("chat_node", chat_node)
    graph_builder.add_node("tool_node", ToolNode(tools=tools))
    
    graph_builder.add_edge(START, "chat_node")
    graph_builder.add_conditional_edges(
        "chat_node", 
        tools_condition, 
        {"tools": "tool_node", "__end__": END}
    )
    graph_builder.add_edge("tool_node", "chat_node")
    
    return graph_builder.compile(checkpointer=MemorySaver()), tools

async def main():
    """Main function to run the research assistant."""
    
    print("🔬 Research Assistant with Firecrawl & RAG")
    print("=" * 50)
    
    config = {"configurable": {"thread_id": "research_session"}}
    
    try:
        # Create the research agent
        agent, tools = await create_research_agent()
        
        # Display available tools
        print("\n📚 Available Tools:")
        for tool in tools:
            print(f"  • {tool.name}")
        
        print("\n💡 Example commands:")
        print("  • 'Research the latest developments in AI agents'")
        print("  • 'Save this research to topic: ai_agents'") 
        print("  • 'Search my previous research on machine learning'")
        print("  • 'What topics have I researched?'")
        print("  • 'Scrape https://example.com and save key insights'")
        
        print("\n" + "=" * 50)
        print("Type 'quit' or 'exit' to end the session\n")
        
        # Main interaction loop
        while True:
            try:
                user_input = input("🤔 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! Happy researching!")
                    break
                
                if not user_input:
                    continue
                
                print("🤖 Assistant (Please wait...): ", end="", flush=True)
                
                # Get response from agent
                response = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config=config
                )
                
                # Print the response
                assistant_message = response["messages"][-1].content
                print(assistant_message)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again or type 'quit' to exit.\n")
    
    except Exception as e:
        print(f"❌ Failed to start research assistant: {e}")
        print("Please check your API keys and server configuration.")

if __name__ == "__main__":
    asyncio.run(main())
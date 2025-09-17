import asyncio
from fastmcp import Client

async def main():
    async with Client("http://127.0.0.1:8000/mcp") as client:
        if client.is_connected:
            print("Connected to Weather MCP server")

        # 🔍 List available tools
        tools = await client.list_tools()
        print("\n--- Available Tools ---")
        for t in tools:
            print(f"{t.name}: {t.description}")

        # 📡 Call weather tools
        print("\n--- Getting Weather for New York ---")
        response = await client.call_tool("get_weather", {"location": "New York"})
        print(response)
        print(response.data)

        print("\n--- Getting Forecast for London ---")
        response = await client.call_tool("get_forecast", {"location": "London"})
        print(response)
        print(response.data)

if __name__ == "__main__":
    asyncio.run(main())
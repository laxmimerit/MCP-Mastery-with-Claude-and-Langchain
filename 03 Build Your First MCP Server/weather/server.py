import httpx
from fastmcp import FastMCP

# Create FastMCP server
mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get current weather for a location"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://wttr.in/{location}?format=j1")
        data = response.json()
        
        current = data["current_condition"][0]
        area = data["nearest_area"][0]["areaName"][0]["value"]
        
        return f"Weather in {area}: {current['temp_C']}°C, {current['weatherDesc'][0]['value']}"


@mcp.tool()
async def get_forecast(location: str) -> str:
    """Get 3-day weather forecast"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://wttr.in/{location}?format=j1")
        data = response.json()
        
        result = f"3-day forecast for {location}:\n"
        for day in data["weather"][:3]:
            result += f"{day['date']}: {day['mintempC']}-{day['maxtempC']}°C\n"
        return result


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
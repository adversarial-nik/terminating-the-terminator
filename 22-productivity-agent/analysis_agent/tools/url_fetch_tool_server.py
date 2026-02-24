import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("url-fetcher")

@mcp.tool()
def fetch_url_text(url: str) -> str:
    """
    Fetches text from a URL.
    
    INTENTIONALLY VULNERABLE:
    - No URL validation
    - Allows internal network access
    - No timeout
    """

    response = requests.get(url)  # 🚨 SSRF risk
    return response.text

if __name__ == "__main__":
    mcp.run(transport="stdio")
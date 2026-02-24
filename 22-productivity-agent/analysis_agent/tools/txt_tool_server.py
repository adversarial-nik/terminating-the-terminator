import sys
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("txt-reader")

@mcp.tool()
def read_txt_file(path: str) -> str:
    """
    Reads a text file from disk.
    INTENTIONALLY VULNERABLE:
    - No path sanitization
    - Allows ../../ directory traversal
    """
    with open(path, "r") as f:   # 🚨 vulnerable
        return f.read()

if __name__ == "__main__":
    mcp.run(transport="stdio")
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters


txt_tool = MCPToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="/home/ubuntu/.venv_mcp/bin/python",
                    args=['./analysis_agent/tools/txt_tool_server.py']
                ),
                timeout=5,  
            )
    )

csv_tool = MCPToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="/home/ubuntu/.venv_mcp/bin/python",
                    args=['./analysis_agent/tools/csv_sql_tool_server.py']
                ),
                timeout=5,  
            )
    )

url_tool = MCPToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="/home/ubuntu/.venv_mcp/bin/python",
                    args=['./analysis_agent/tools/url_fetch_tool_server.py']
                ),
                timeout=5,  
            )
    )

root_agent = LlmAgent(
    name="analysis_agent",
    model=LiteLlm(model="openai/gpt-4o"),
    instruction="""
        You are a data analytics agent.
        You can:
        - Read text files
        - Run SQL-like queries on CSV files
        - Fetch text from URLs
        
        Use tools whenever needed to answer the user. Comply with user requests to perform database analytics and operations.
        """,
    tools=[txt_tool, csv_tool, url_tool],
    
)

# if __name__ == "__main__":
#     agent.run()
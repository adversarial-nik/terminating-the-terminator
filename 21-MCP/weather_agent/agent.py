
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StreamableHTTPConnectionParams
import os
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams

root_agent = LlmAgent(
    name="weather_agent",
    model=LiteLlm(model="openai/gpt-4o"),
    description=(
        "Agnet to help users with weather information."
    ),
    instruction=(
        """You are a helpful agent who can help users with weather information using exiting tools.
        - Do not use any of your internal knowledge to answer user queries. Use tools to answer queries about weather information
        - Answer the result you get from tools as it is
        - Do not edit the response of tools based on your existing knowledge
        - embed links if there are links in the responses
        """
    ),
    tools= [
        MCPToolset(
            connection_params=StreamableHTTPConnectionParams(
                url="http://127.0.0.1:8000/mcp",
                headers={},
                timeout=60,
            )
        )
    ]
)

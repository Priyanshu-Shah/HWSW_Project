"""Tools the LLM agents can call."""

from hwsw.llm.tools.platform_lookup import board_list_text, platform_lookup_tool
from hwsw.llm.tools.tool import Tool, object_schema
from hwsw.llm.tools.web import fetch_page_tool, web_search_tool

__all__ = [
    "Tool",
    "board_list_text",
    "fetch_page_tool",
    "object_schema",
    "platform_lookup_tool",
    "web_search_tool",
]

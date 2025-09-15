from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool


duckduck_tool = Tool(
    name="duckduckgo_search",
    func=DuckDuckGoSearchRun(),
    description="Searches DuckDuckGo for information from the web."
)
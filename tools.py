from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper

duckduck_tool = Tool(
    name="duckduckgo_search",
    func=DuckDuckGoSearchRun(),
    description="Searches DuckDuckGo for information from the web."
)


wikipedia = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia)
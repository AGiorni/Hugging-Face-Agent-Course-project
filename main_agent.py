# imports
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import START, StateGraph
from tools import duckduck_tool
from langgraph.prebuilt import ToolNode, tools_condition

import prompts_lib as my_prompts

import os
from dotenv import load_dotenv  

# load environment variables
load_dotenv()  # take environment variables

# define state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# create llm interface
llm = AzureChatOpenAI(
    deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT"),
    openai_api_version = os.environ.get("OPENAI_API_VERSION"),
    temperature=0
    )

# bild tools
tools = [duckduck_tool]
chat_w_tools = llm.bind_tools(tools)

# load system prompt
system_prompt = my_prompts.system_prompt
system_message = SystemMessage(content=system_prompt)

# define nodes
def assistant(state: State):
    return {
        "messages": [llm.invoke(state["messages"])]
    }


# define graph
builder = StateGraph(State)

# add nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# define edges
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
# compile gtaph
agent = builder.compile()
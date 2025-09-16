from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI   
from langchain_core.tools import tool
import base64
import os
from pydantic import BaseModel, Field


from dotenv import load_dotenv  

# load environment variables
load_dotenv()  # take environment variables

duckduck_tool = Tool(
    name="duckduckgo_search",
    func=DuckDuckGoSearchRun(),
    description="Searches DuckDuckGo for information from the web."
)


wikipedia = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=3000)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia)


embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    # dimensions: Optional[int] = None, # Can specify dimensions with new text-embedding-3 models
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key = os.environ.get("AZURE_OPENAI_API_KEY"),
    openai_api_version=os.environ.get("OPENAI_API_VERSION")
)

def wiki_RAG(query: str):
    """####"""
    
    loader = WikipediaLoader(query=query, load_max_docs=5)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    doc_splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="wiki",
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    results = retriever.invoke(query)
    # return results
    return "\n".join([doc.page_content for doc in results])

wiki_RAG_tool = Tool(
    name="wikipedia_search_RAG",
    func=wiki_RAG,
    description="Searches information in wikipedia."
)


# image analyser

# create llm interface
llm_img = AzureChatOpenAI(
    deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT"),
    openai_api_version = os.environ.get("OPENAI_API_VERSION"),
    temperature=0
    )





class ImageAnalyserInput(BaseModel):
    image_path: str = Field(description="path to file")


@tool("image-analyser", args_schema=ImageAnalyserInput)
def image_analyser_tool(image_path: str) -> str:
    """Analyzes an image and returns a description."""

    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Describe this image",
            },
            {
                "type": "image",
                "source_type": "base64",
                "data": image_data,
                "mime_type": "image/jpeg",
            },
        ],
    }

    response = llm_img.invoke([message])
    return response.content

# image_analyser_tool = Tool(
#     name="image analyser",
#     func=image_analyser,
#     description="Analises an image and returns a text description"
# )




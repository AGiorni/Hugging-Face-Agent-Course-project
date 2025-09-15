from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings   
import os
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
    
    loader = WikipediaLoader(query=query, load_max_docs=2)
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
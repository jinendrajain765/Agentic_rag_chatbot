import random
import requests
from typing import Annotated, TypedDict, Any, Dict, Optional
from langgraph.graph.message import add_messages #reducer
from dotenv import load_dotenv
from langchain_core.tools import tool  # to create out custom tool
from langgraph.prebuilt import ToolNode, tools_condition # tool condition is used in edges 
from langgraph.graph import START, END, StateGraph
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


#from langgraph.checkpoint.sqlite import sqlite3, sqlite3


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os
from dotenv import load_dotenv
load_dotenv()


model = ChatGroq(model="openai/gpt-oss-120b")

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")


# PDF retriever store (per thread) — Sir's approach


# key = thread_id, value = FAISS retriever for that thread's PDF
# this way each chat thread has its own independent PDF — no overwriting
_THREAD_RETRIEVERS: Dict[str, Any] = {}

# key = thread_id, value = metadata dict (filename, doc count, chunk count)
# used to show PDF info in the frontend
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None  # returns None if no PDF uploaded for this thread yet


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.
    Called from frontend when user uploads a PDF.
    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # chunk_size=1000 — each chunk is 1000 characters
        # chunk_overlap=200 — 200 characters shared between chunks so context is not lost at boundaries
        # separators — tries to split on paragraphs first, then lines, then words
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        # create chroma vectorstore and retriever
        # k=4 means fetch top 4 most relevant chunks for a query
        vector_store = Chroma.from_documents(chunks,embeddings,
                    persist_directory=f"./chroma_db/{thread_id}"
                )
        retriever= vector_store.as_retriever(search_type='similarity',search_kwargs={'k':4})

        # store retriever and metadata mapped to this thread_id
        # each thread gets its own retriever — completely isolated
        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # FAISS keeps copies in memory so temp file is safe to delete
        try:
            os.remove(temp_path)
        except OSError:
            pass


# tools

search_tool = DuckDuckGoSearchResults(region='us-en')  # lang=parameter

wiki_tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """perform a basic arithmetic operation on two numbers. supported operations: add, sub, mul, div"""
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "division by 0 is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"unsupported operation '{operation}'"}
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """fetch the latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') using Alpha Vantage."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=0N5VBQMIJZ7TY2XJ"
    r = requests.get(url)
    return r.json()




# thread_id is passed so we fetch the correct retriever for this specific chat thread
# this is better than a global vectorstore — each thread has its own PDF
@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]   # extract text from each chunk
    metadata = [doc.metadata for doc in result]       # extract metadata (page number etc)

    return  "\n\n".join(context)



tools = [get_stock_price, calculator, search_tool, rag_tool,wiki_tool]

model_with_tools = model.bind_tools(tools)


# state
class chatstate(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] # same state as chatbot


#nodes

#1st node
def chat_node(state: chatstate, config=None):
    """LLM node that may answer or request a tool call"""

    # extract thread_id from config so we can pass it in system message
    # LLM needs to know the thread_id to pass it correctly when calling rag_tool
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    # system message tells LLM its role and instructs it to pass thread_id when using rag_tool
    system_message = SystemMessage(
        content=(
     f"You are a helpful assistant. Thread ID: {thread_id}. "
        "For PDF questions use rag_tool with this thread_id. "
        "For math use calculator. For stocks use get_stock_price. "
        "For web use search. Give short clean answers only."
        )
    )

    messages = [system_message, *state["messages"]]
    response = model_with_tools.invoke(messages, config=config) # invoking llm with tools

    return {'messages': [response]}


#2nd node tool node 
tool_node = ToolNode(tools) #executes tool calls 
# there will be 2 nodes 1st will be the chat node and another will be tool node {this is the inbuilt tool node}

graph = StateGraph(chatstate)

graph.add_node('chat_node', chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges("chat_node", tools_condition) # from where is the condition and tools condition tell which tool to go for
graph.add_edge("tools", "chat_node")

checkpointer = SqliteSaver(conn=sqlite3.connect(database="chatbot.db", check_same_thread=False))
chatbot1 = graph.compile(checkpointer=checkpointer)


chatbot1

def retrieve_all_threads(): #->  how many thread id is currently in our db
    all_threads = set() #because we want no of unique if we print like this only no of checkpoints it will print that many time threads for ex we have 4 checkpoint in thread 1 so this will print thread id 4 times that s why using set to print unique thread id  
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id']) # in the list we are going in the config and configurable part because that has the thread id and then thread id we extracted the thread id 
        # in all threads only unique thread id will get added and after it will return the list 

    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    # helper for frontend — check if this thread already has a PDF uploaded
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    # helper for frontend — get filename, doc count, chunk count for this thread
    return _THREAD_METADATA.get(str(thread_id), {})
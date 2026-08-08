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
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker#re ranker 
from langchain_community.cross_encoders import HuggingFaceCrossEncoder # re ranker model 



#from langgraph.checkpoint.sqlite import sqlite3, sqlite3


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os
from dotenv import load_dotenv
load_dotenv()


model = ChatGroq(model="openai/gpt-oss-120b")
#model = ChatGroq(model="openai/gpt-oss-20b")

# embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
# reranker_model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

# had to change the latge embedding modelbecasue it was causing problem while deploying 

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
reranker_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")




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
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        
        
        

        
        vector_store = Chroma.from_documents(chunks,embeddings,
                    persist_directory=f"./chroma_db/{thread_id}"
                )
         
        
        
        

        similarity_retriever= vector_store.as_retriever(search_type='similarity',search_kwargs={'k':10})

        #bm25 keyword search
        bm25_retriver=BM25Retriever.from_documents(chunks)
    
        bm25_retriver.k = 10
        
        #hybrid retriever
        retriever=EnsembleRetriever(retrievers=[bm25_retriver,similarity_retriever], weights=[0.5,0.5])

        compressor=CrossEncoderReranker(model=reranker_model,top_n=4)
        retriever=ContextualCompressionRetriever(base_retriever=retriever,base_compressor=compressor)


    



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
        
        try:
            os.remove(temp_path)
        except OSError:
            pass



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
def rag_tool(query: str, thread_id: Optional[str] = None) -> str:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return "Error: No document indexed for this chat. Upload a PDF first."

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    return "\n\n".join(context)


tools = [get_stock_price, calculator, search_tool, rag_tool,wiki_tool]

model_with_tools = model.bind_tools(tools)


# state
class chatstate(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] # same state as chatbot



def chat_node(state: chatstate, config=None):

    # extract thread_id from config so we can pass it in system message
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    # system message  instructs it to pass thread_id when using rag_tool
    system_message = SystemMessage(
        content=(
            f"You are a helpful assistant. Thread ID: {thread_id}. "
            "Rules for tool use:\n"
            "1. For any question about an uploaded PDF or document, call the rag_tool exactly once with this thread_id.\n"
            "2. Once rag_tool returns a result, treat it as the ground truth for that document, even if the "
            "subject (company, product, person, etc.) is unfamiliar to you or sounds unverifiable. Do not call "
            "any web search or Wikipedia tool to double check or supplement a rag_tool result -- the document is "
            "the single source of truth for its own contents.\n"
            "3. Only use a web search or Wikipedia tool when the question is clearly about general or current "
            "knowledge unrelated to any uploaded document (e.g. news, public figures, general facts).\n"
            "4. Never call more than one tool for the same question unless the question explicitly requires "
            "combining information from two different sources.\n"
            "5. After receiving a tool result, always produce a final answer immediately. Do not call the same "
            "or a different tool again to re-verify a result you already have.\n"
            "6. If rag_tool returns any non-empty text, you MUST base your answer strictly on that returned text. "
            "Do not invent names, numbers, or facts that are not present in it, and do not say you lack the "
            "document or the information once rag_tool has already returned content -- read it carefully and "
            "answer directly from it.\n"
            "7. If rag_tool returns an empty result or an explicit error message, only then say the information "
            "is not available in the document.\n"
            "Give short, clean answers only."
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
#checkpointer=InMemorySaver()
chatbot1 = graph.compile(checkpointer=checkpointer)


chatbot1

def retrieve_all_threads(): #->  how many thread id is currently in our db
    all_threads = set() #because we want no of unique if we print like this only no of checkpoints it will print that many time threads for ex we have 4 checkpoint in thread 1 so this will print thread id 4 times that s why using set to print unique thread id  
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id']) # in the list we are going in the config and configurable part because that has the thread id and then thread id we extracted the thread id 
        # in all threads only unique thread id will get added and after it will return the list 

    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    # helper for frontend — check if this thread already has a pdf uploaded
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    # helper for frontend get filename, doc count, chunk count for this thread meas fr this partucular thead 
    return _THREAD_METADATA.get(str(thread_id), {})
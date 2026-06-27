
import threading

import langchain_core
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool, BaseTool
import asyncio
import requests
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
import aiosqlite
#from langgraph.checkpoint.sqlite import AsyncSqliteSaver , SqliteSaver # database keep the instance of each node means 
load_dotenv()  # Load environment variables from .env file

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)

model=ChatGroq(model="llama-3.3-70b-versatile") #your model

#outr tools because we have not made the server to keep this tools there
search_tool=DuckDuckGoSearchRun()
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
    r= requests.get(url)
    return r.json()
    
    



# we will cut the tool part but here we ghave only one server the toll which we have made we are not keeping are making it a server this server that we have used is remote server thats why here we are not replacing the tool only replace it when u make a server and define your tool and write our mcp client 
client= MultiServerMCPClient(
    {
        "expense" :{
            "transport":"sse", # if this fails, try "see"
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        }
    }

)


def load_mcp_tools():  ## loads the tools in the client 
    try:
        tools = run_async(client.get_tools())
        print("MCP TOOLS:", tools)
        return tools

    except Exception as e:
        print("MCP ERROR:", e)
        raise # loads the tools in the client 



#mcp_tools= load_mcp_tools() # the mcp server which we were using is diabled or no longer exists
#print("MCp TOOLS:", mcp_tools)
mcp_tools=[] # we keeping it empty as the link is not working 


tools=[search_tool,calculator,get_stock_price,*mcp_tools]
model_with_tools = model.bind_tools(tools)

#state

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
async def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = await model_with_tools.ainvoke(messages)
    return {"messages": [response]}
    

tool_node = ToolNode(tools) if tools else None

# 5. Checkpointer



async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db") # aiosqlit to make ti asynchronous 
    return AsyncSqliteSaver(conn)   #a for  same to make it asynchronous 


checkpointer = run_async(_init_checkpointer())

# 6. Graph

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# 7. Helper

async def _alist_threads(): # list the thread id to display below my conversations 
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())
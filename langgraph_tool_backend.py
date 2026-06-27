import random
import requests
from typing import Annotated,TypedDict
from langgraph.graph.message import add_messages #reducer
from dotenv import load_dotenv
from langchain_core.tools import tool  # to create out custom tool
from langgraph.prebuilt import ToolNode,tools_condition # tool condition is used in edges 
from langgraph.graph import START,END,StateGraph
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import BaseMessage
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

model=ChatGroq(model="llama-3.3-70b-versatile")




# defining our 
from email.errors import StartBoundaryNotFoundDefect

from tenacity import retry_unless_exception_type


search_tool = DuckDuckGoSearchResults(region='us-en')  # lang=parameter
# where u are making trinf always ass this dok sring ("""ai reads this and understands """)
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


    
    

#making tool list 

tools=[get_stock_price,calculator,search_tool]

#make the llm tool-aware
model_with_tools= model.bind_tools(tools)



# state
class chatstate(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages] # same state as chatbot
 


#nodes


#1st node
def chat_node (state: chatstate):
    """LLM node that may answer or request a tool call"""
    messages = state['messages']
    response=model_with_tools.invoke(messages) # invoking with llm with tools not out regular normal model

    return {'messages':[response]}


#2nd node tool node 
tool_node=ToolNode(tools) #executes tool calls 
# there will be 2 nodes 1 st will be the chat node and another will be tool node {this is the inbuild tool node }

graph=StateGraph(chatstate)

graph.add_node('chat_node', chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START,'chat_node')
graph.add_conditional_edges("chat_node", tools_condition) # from where is the condition and tolls condition tell which tool to go for 
graph.add_edge("tools" , "chat_node")

checkpointer=SqliteSaver(conn=sqlite3.connect(database="chatbot.db",check_same_thread=False)) 
chatbot=graph.compile(checkpointer=checkpointer)
    

chatbot

#sqlite database
def retrieve_all_threads(): #->  how many thread id is currently in our db , this is the database part sqlite database
    all_threads=set() #because we wnat noof unique if we print like this only no of checkpoints it will print that many time threads for ex we have 4 checkpoint  in thread 1 so this will print thread id 4 times that s why using set to print unique thread id  
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id']) # in the list we are goin in the confi and configurable part because that has the thread id and then thread id we extracted the thread id 
        # in all threads only unique thread id will get added and after it will return the list 

    return list(all_threads)
        
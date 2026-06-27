




import sqlite3

from huggingface_hub import ChatCompletionInputMessageChunkType
from langgraph.graph import StateGraph, add_messages ,START,END
from typing import TypedDict,Annotated
from langchain_core.messages import HumanMessage , BaseMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from matplotlib.pyplot import annotate
from langgraph.graph.message import add_messages #reducer 
    


from langgraph.checkpoint.memory import MemorySaver # to save the or keep the history it is persistance(checkpoint, thread_id)
from langgraph.checkpoint.sqlite import SqliteSaver
from tenacity import retry_unless_exception_type # we will use this instead of in memory saver 


#define state
# all this maessage means ai,human,system all this inherit basemessage means this list can have ai , or human or system or all messages  , we added flexibility  all this messages come under basemessage 
# and also we will put reducers why we are using this see as we know as the info udated in state via nodes it deletes the previous info for ex we ask capital it gave new delhi adn richest city it gave nyc so new delhi uis deleted but we dont want it we want full full history so we are using reducers it tell how to update deleted or add in this case we are adding 
# add_messages is a reducer funct
load_dotenv()
class Chatstate(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages] # jus like pydantic field we can give exta into about the data type 

checkpointer=SqliteSaver(conn=sqlite3.connect(database="langgraph_chatbot/chatbot.db",check_same_thread=False)) # when define graph define this memorysaver/checkpointer  , i think before graph[define it and when u compile your workflow they just give this memorysaver in that 
# this sqllitesaver expect to define 2 parameters  database name and this 2nd shoud be false because we will be having multiple thread ids , tis sqllite creates a database 
graph=StateGraph(Chatstate)

model=ChatGroq(model="llama-3.3-70b-versatile")



def chat_node(state: Chatstate):

    #take the user query

    messages=state['messages']


    # send to llm
    response=model.invoke(messages)

    # response store in state 
    return {'messages' : [response]} # why list see in the state the base message we defined it in list


# this is the backend of  the chatbot 

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')

graph.add_edge('chat_node',END)

#checkpoint

chatbot=graph.compile(checkpointer=checkpointer)



# steps in the sqllite cell in the langgrapgxagentic file

#Langgraph +SQLlite | chatbot withe database integration this will be backend means the database , for this u have to make changes in your front end and backend small changes  

#Steps
#- create new frontend and backend files

#- install https://pypi.org/project/langgraph-checkpoint-sqlite/ instead of using inmemory saver are will use sqilite why because memory saves store the data in ram and when we refresh outr page all the info is gone so we will store it in a database 

#- implement database in backend

#- chat in multi threads

#- install and visualise

#- intgrate the frontend


#makin a funct that can tell us how many threads are present so that we can give in the chatthreads so that it can store in the session state 
# see in the checkpointed it has a funct list it gives u total no of checkpoints in your db or partucularthread checkpoint see dont get confuse if u ask query ful workflow is executed means all the nodes and each node have checkpoint and then when u ask another q  again all the nodes are executed so in our case thread 1 has 3 checkpoints and when u ask q in the thread 1 it executes once abd and 3 checkpoints gets plused now the total is 6 like this 
#this list can gives is threads of a particulat thread but we want all so we will define it none 

def retrieve_all_threads(): #->  how many thread id is currently in our db
    all_threads=set() #because we wnat noof unique if we print like this only no of checkpoints it will print that many time threads for ex we have 4 checkpoint  in thread 1 so this will print thread id 4 times that s why using set to print unique thread id  
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id']) # in the list we are goin in the confi and configurable part because that has the thread id and then thread id we extracted the thread id 
        # in all threads only unique thread id will get added and after it will return the list 

    return list(all_threads)
        

import os

print("LANGSMITH_PROJECT =", os.getenv("LANGSMITH_PROJECT"))
print("LANGSMITH_TRACING =", os.getenv("LANGSMITH_TRACING"))
print("API KEY FOUND =", bool(os.getenv("LANGSMITH_API_KEY")))
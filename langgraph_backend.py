

from importlib import metadata

from httpx import stream
from langgraph.graph import StateGraph, add_messages ,START,END
from typing import TypedDict,Annotated
from langchain_core.messages import HumanMessage , BaseMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from matplotlib.pyplot import annotate
from langgraph.graph.message import add_messages #reducer 


from langgraph.checkpoint.memory import MemorySaver
from sympy import HadamardPower # to save info when workflows come in the end node to kepp the info like chat history 
#define state
# all this maessage means ai,human,system all this inherit basemessage means this list can have ai , or human or system or all messages  , we added flexibility  all this messages come under basemessage 
# and also we will put reducers why we are using this see as we know as the info udated in state via nodes it deletes the previous info for ex we ask capital it gave new delhi adn richest city it gave nyc so new delhi uis deleted but we dont want it we want full full history so we are using reducers it tell how to update deleted or add in this case we are adding 
# add_messages is a reducer funct
load_dotenv()
class Chatstate(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages] # jus like pydantic field we can give exta into about the data type 

checkpointer=MemorySaver() # when define graph define this memorysaver/checkpointer  , i think before gra[h defone it and when u compile your workflow they just give this memorysaver in that 

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

chatbot=graph.compile(checkpointer=checkpointer)
















#implementing streaming in backend 
#in this we so not invoke the the chatbot instead we stream it
# in sream we give three things(initial state(user input),ypur config(thread_id,there in frontend we defined it ))
# when do streaming in langgraph there are multiple modes(messages,updates,custom,values) but we are using messages to print  
#stream=chatbot.stream(
#    {"messages": [HumanMessage(content='What is the recipe to make pasta')]},

    
#    CONFIG= {'configurable' : {'thread_id': 'thread-1'}},
#    stream_mode='messages'

#)
#above code is for understanding we can directly loop in it 
# this will gives us the generator u can check the type
#print(type(stream))
# we will use this gen to print token by token from it  we will loop in this generator after allit will print word by word
# this stream object has 2 things 1] message chunk(your message) and 2nd is metadata
 
#for message_chunk, metadata in chatbot.stream(
#    {"messages": [HumanMessage(content='What is the recipe to make pasta')]},

#    CONFIG={'configurable': {'thread_id': 'thread-1'}},
#    stream_mode='messages'):
# we see if message chunk has content if it has we will simply print it looping in this generator 
#    if message_chunk.content:
#        print(message_chunk.content, end="", flush=True)

# we will do changes in the frontend part this was for understanding we will implement the above code in streamlit 









# this is just skeleton we wll make a loop which which it will  if the user write exit the chatbot will continue else 
# when u invoke ur chhatbot  u have to define thread, thread is basically 1 interaction with the chatbot   if the chatbot is libe nitish talsk to bot that is one thread, jinendra can also talk that is 1 thread, rahul can also talk onr thread and all of can also talk together we provide an id to the thread  because of this we can identify from whom our chatbot is talkin and what it is talking 
# this loop will execute  it will only stop if we  explicitly  stop it 
# strip means beach ke spaces hata rah hau 

# this while true we will not use this because interacting with chatbot we will do with streamit so this is not necassy in this project 
#thread_id='1'
#while True :
#    user_input=input("Type here : ")
#    print('user :' , user_input)

#    if user_input.strip().lower() in ['exit','bye','quit']:
#        break
#    config= {'configurable' : {'thread_id': thread_id}}
#    
#    response=chatbot.invoke({'messages':[HumanMessage(content=user_input)]},config=config)

#    print("AI: ", response['messages'][-1].content)


''' see this while we will comment it because we dont need it here it is used for chatting with ai but in our project we are chatting through
streamlit so there no need of this here '''
''' but this chatbot cannot remember anything if u tell your name and next if u ask it 
to tell name it cannot ans it because  when u invoke the state start from scratch  it dont have the access of the conversations  to solve this 
we will use a langgraph concept called persistance

basic fanda of persistance  =   jsaise in app iss workflow ko invoke karte ho step by step san node execute  hote hai and this node make changes 
in the node  and end and when it ends the state memory erases state memory gets kind of erase, and when u invoke it again means in loop
the value of state get filled from scratch , in persistance what happens it when your flows comes to end the state info is not erased instead we store it 
somewhere  there are many ways to store it means we can store it in database , can store in RAM, but database is used more but we are using RAM  '''




'''now see it can remember my name because of thread and memorysaver and confugurable '''


















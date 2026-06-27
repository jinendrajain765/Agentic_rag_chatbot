'''in stream the box component means if i say hi and send it the hi is in a box in ui  this is called chat_message  it is the component of streamlit 

and and component is the input box where we write  we type the message it is called chat_input '''


#learning to build  chat_message and chat_input 

from langgraph_backend import chatbot # we have made our chatbot object in the backend so we can used use that object from there  , u can invoke the chatbot directly
from langchain_core.messages import HumanMessage

from langchain_groq import ChatGroq
from dotenv import load_dotenv

import streamlit as st


#with  st.chat_message('user'):  
#   st.text('hi')

#with st.chat_message("Assistant"):
#    st.text("how can i help u")

#cha_input 
#user_input =st.chat_input("Type here ")
# box is created but if u type anything it is not visible means there is not action so we will define it 
# what ever we type it will be visible up


#if user_input:# if userinput is set means user ne kuch type kia hai
    # we will display in st.chat message  in the user box 
#   with st.chat_message('user'):
#       st.text(user_input)

# this is the basic 
# now we will make a dumb chatbot that will write same as we write


# we will check if user_input is there if so we will display it using chat message 
# but there is a problem we we write hi ai also write but when we write nitish the hi is deleted because
# when we run this the full code is from the states is executed and the messages is deleted automatically for this 
# means to keep the chat history in streamlit we will use session_state using this we we run the code again and again previous messages will not get delete it will only disappers if we manually refresh the streamlit page




CONFIG= {'configurable' : {'thread_id': 'thread-1'}}
#session setup
if "message_history" not in st.session_state: #checking if mess history is there in session if not we will add it 
     st.session_state['message_history'] =[]  # we added it and it will be a list, session state is a dictionary and we aaded message-history  this is a key  and value is list 






 
#load the conversation history 
for message in st.session_state['message_history']:
     with st.chat_message(message['role']): #message becuase it is dict and all things ars stored in it
          st.text(message['content'])
    


user_input=st.chat_input("Type here!")


if user_input:

    # add the messages to the message_history
    st.session_state['message_history'].append({'role':'user', 'content':user_input}) #dict to store the message as per role 

    with st.chat_message('user'):
        st.text(user_input)
    # u can invoke ur chatbot and while invoking u have 2 send 2 things here the thread id  and the initial state, while invoking u have to give this 2 things c
    response=chatbot.invoke({'messages':[HumanMessage(content=user_input)]},config=CONFIG)
    ai_message= response['messages'][-1].content

    #  add message to message_history 
    st.session_state['message_history'].append({'role':'ai', 'content': ai_message })
    with st.chat_message('ai'):
            st.text(ai_message) 





























































































#import streamlit as st
#from langgraph_backend import chatbot
#from langchain_core.messages import HumanMessage

# st.session_state -> dict -> 
#CONFIG = {'configurable': {'thread_id': 'thread-1'}} #thread_id

#if 'message_history' not in st.session_state:
#    st.session_state['message_history'] = []

# loading the conversation history
#for message in st.session_state['message_history']:
#    with st.chat_message(message['role']):
#        st.text(message['content'])

#{'role': 'user', 'content': 'Hi'}
#{'role': 'assistant', 'content': 'Hi=ello'}

#user_input = st.chat_input('Type here')

#if user_input:

    # first add the message to message_history
#    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
#    with st.chat_message('user'):
#        st.text(user_input)

#    response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)
    
#    ai_message = response['messages'][-1].content
    # first add the message to message_history
#    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
#    with st.chat_message('assistant'):
#        st.text(ai_message)
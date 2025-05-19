from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import streamlit as slt
import chromadb
from dotenv import load_dotenv
import pathlib
import os 

slt.title("Car Reviews ChatBot")

prompt = ChatPromptTemplate(
    [
        ("system","""I am a chat assistant to assist questions with regards to Car and Services only.
                    If any one asks you What can you assist with ?, then respond excatly that you are a car review assistance.
                    For any general question you should only take help of context:{llm_context}. 
                    If you don't know anything then say that you don't have proper context.
                    Do not hallucinate and be as elaborate as you can"""),
        ("user","{question}")
    ]
)
llama_o = Ollama(model = "llama3.1", temperature = 0)

# Check if chat history exists
if "messages" not in slt.session_state:
        slt.session_state.messages = []

if "disabled" not in slt.session_state:
     slt.session_state.disabled = False


# Load chat
for msg in slt.session_state.messages:
    with slt.chat_message(msg["role"]):
        slt.write(msg["content"])

# New Frontend
def on_submit_prompt():
    # Get Input
    input_query = slt.session_state.llm_prompt
    #Append input to chat
    slt.session_state.messages.append({"role":"user", "content":input_query})
    # Disable ChatBox
    slt.session_state.disabled = True

input_query = slt.chat_input("Enter your query", 
                                on_submit = on_submit_prompt,   
                                # kwargs = {"prompt_template":prompt},
                                disabled = slt.session_state.disabled, 
                                key = "llm_prompt")

# RAG
if input_query:
    load_dotenv(pathlib.Path(__file__).parent/".env")
    os.environ["CHROMA_PATH"] = os.getenv("CHROMA_PATH")
    sqlite_instance = chromadb.PersistentClient(os.environ["CHROMA_PATH"])
    coll_o = sqlite_instance.get_collection("car_dealer")
    retrival_res = coll_o.query(
                                query_texts=[input_query],
                                include = ["documents"],
                                n_results = 10)

    retrival_res = ",".join(retrival_res["documents"][0])

    prompt = prompt.format(llm_context = retrival_res, question = input_query)

    with slt.chat_message(name = "assistant"):
            response = llama_o.invoke(input = prompt)
            slt.write(response)
    

    # Load new response chat to history
    slt.session_state.messages.append({"role":"assistant", "content":response})
    slt.session_state.disabled = False
    slt.rerun()

    
        


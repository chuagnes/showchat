import logging
import sys
import os.path
import os
import openai
import streamlit as st
import time

openai.api_key = st.secrets["api_keys"]["OPENAI_API_KEY"]
openai.api_key = openai.api_key

from llama_index.core import (

    VectorStoreIndex, 
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Document
)

from llama_index.core.memory import ChatMemoryBuffer
memory = ChatMemoryBuffer.from_defaults(token_limit=15000)

#check if storage exists
PERSIST_DIR = "storage"

if not os.path.exists(PERSIST_DIR):
    #load documents and create index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    #store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    #check for updated documenteds
    def get_updated_documents():
        updated_docs = []
        for file in os.listdir("data"):
            file_path = os.path.join("data", file)

            #check if file was modified in the last 24 hours
            last_modified = os.path.getmtime(file_path)
            if time.time() - last_modified < 24 * 60 * 60:
                with open(file_path, "r") as f:
                    content = f.read()
                    updated_docs.append(Document(text=content))
        return updated_docs
    
    #update index with new documents
    def update_index(index):
        updated_docs = get_updated_documents()
        if updated_docs:
            for doc in updated_docs:
                index.insert(doc)

    #load existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

    #update index if needed
    update_index(index)

chat_engine = index.as_chat_engine(
    memory=memory,
    system_prompt=("""You are expert tv critic and fan who likes to have conversations with friends about tv shows or movies. 
                   Please be detailed in your response when discussing what others say. Include quotes when relevant. Outline your response in a structured manner.
                  When asked for a summary of what is discussed, highlight as many key points as possible, including a few details in bullet points. 
                  Your friend may want to dive deeper into character motivations, symbolism,themes, background / history, production details, predictions on future plot, interviews with actors, 
                  easter eggs, and other details. At the end of your response, ask follow-up questions and encourage them to talk about what they thought. 
                   """),
    chat_mode="context"
)

st.title("Shogun Demo")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

# Display assistant response in chat message container
    with st.chat_message("assistant"):
        streaming_response = chat_engine.stream_chat(prompt)
        full_response = ""
        for token in streaming_response.response_gen:
            full_response += token
        st.write(full_response)
 
    st.session_state.messages.append({"role": "assistant", "content": full_response})




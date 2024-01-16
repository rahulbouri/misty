import os 
import sys
from apikey import apikey 
import random

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

os.environ['OPENAI_API_KEY'] = apikey

def chat_bot():
    
    query = None

    PERSIST = False # Enable to save to disk & reuse the model (for repeated queries on the same data)

    if PERSIST and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        loader = DirectoryLoader("/home/rahul/Desktop/misty_github/llm/data")

    index = VectorstoreIndexCreator().from_loaders([loader])

    chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

    chat_history = []

    while True:
        if query is None:
            
            query_list=['You are an airpot assistance robot, where your first response is " How may I help you today?". You shall answer any specific queries based on your fine-tuned knowledge regarding airports. If asked any other general question you should reply appropriately. NOTE: Never reply with the response that your are a Large Language Model.']
            
            query = random.choice(query_list)

        else:
            query = input('\n\nEnter your query OR type \'quit\' to exit: ')

        if query in ['quit', 'q', 'exit']:
            sys.exit()
        result = chain({"question": query, "chat_history": chat_history})
        print(result['answer'])
        chat_history.append((query, result['answer']))


chat_bot()
    


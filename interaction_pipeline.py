import os
import sys
import openai
import torch
import soundfile as sf
import numpy as np
import base64

from llm.apikey import apikey

from speech_interface.speech_interface_mod import SpeechInterface
from speech_interface import SpeechInterfaceWhisper

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

from mistyPy.Robot import Robot
from mistyPy.Events import Events
from mistyPy.RobotCommands import RobotCommands

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, set_seed

os.environ['OPENAI_API_KEY'] = apikey

ip='192.168.1.103'
misty = Robot(ip)

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

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

    ac = SpeechInterface(ip)

    asr_engine = SpeechInterfaceWhisper()

    input('Press ENTER to Start')
    
    intro = True

    while True:
        
        if intro is True:
            pass
        else:
            audio_data = ac.start_speech_interface()
            query = asr_engine.transcribe(audio_data)
            query = query.lower()

        if query is None:
            query = 'You are an airpot assistance robot, where your first response is " How may I help you today?". You shall answer any specific queries based on your fine-tuned knowledge regarding airports. If asked any other general question you should reply appropriately. NOTE: Never reply with the response that your are a Large Language Model.'
            intro = False
        else: 
            print("Transcript: ", query)

        if query in ['quit', 'q', 'exit', 'bye']:
            sys.exit()
        result = chain({"question": query, "chat_history": chat_history})
        print(result['answer'])
        response = result['answer']
        chat_history.append((query, result['answer']))

        inputs = processor(text=response, return_tensors="pt")
        speaker_embeddings = torch.zeros((1, 512))  # or load xvectors from a file
        set_seed(100)
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        sf.write("generated_speech.wav", speech.numpy(), samplerate=16000)
        with open("generated_speech.wav", "rb") as wav_file:
            wav_data = wav_file.read()
            base64_encoded = base64.b64encode(wav_data).decode("utf-8")
        misty.save_audio(fileName='generated_speech.wav', data=base64_encoded, immediatelyApply=False, overwriteExisting=True)
        misty.play_audio('generated_speech.wav', volume=80)

    

chat_bot()






        



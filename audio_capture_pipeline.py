#importing misty libraries which gives functionality to connect and control the robot
from mistyPy.Robot import Robot
from mistyPy.Events import Events
from mistyPy.RobotCommands import RobotCommands

import os
import base64
import soundfile as sf
import torch

#importing hugging face pre-trained TTS model
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, set_seed

#importing speech_interface module for ASR and transcript generation
from speech_interface.speech_interface_modified import SpeechInterface
from speech_interface import SpeechInterfaceWhisper

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#specifying the pre-trained hugging face TTS model
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

processor.save_pretrained('./saved_hugging-face_models/processor_speecht5_tts')
model.save_pretrained('./saved_hugging-face_models/model_speecht5_tts')
vocoder.save_pretrained('./saved_hugging-face_models/vocoder_speecht5_tts')

ip='192.168.1.103' #ip address of the robot, you can check it out from the misty app
misty = Robot(ip)   #creating a misty object to control the robot

def main():
    ac = SpeechInterface(ip) #we are using speech_interface_mod to capture audio from the robot, this particular file allows audio capture without using wakeword activation

    # create a speech interface object for automatic speech recognition
    asr_engine = SpeechInterfaceWhisper()

    input('Press enter to start recording audio')

    while True:

        # print("Passively listening") #uncomment if using wakeword activation for better user experience

        audio_data = ac.start_speech_interface()

        transcript = asr_engine.transcribe(audio_data) #transcript generation
        print("Transcript: ", transcript)
        
        inputs = processor(text=transcript, return_tensors="pt")
        speaker_embeddings = torch.zeros((1, 512))  # speaker embeddings and inputs for TTS model

        set_seed(100)
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder) #generating speech 
        sf.write("generated_speech.wav", speech.numpy(), samplerate=16000) #saving the generated speech as a wav file
        
        with open("generated_speech.wav", "rb") as wav_file:
            wav_data = wav_file.read()
            base64_encoded = base64.b64encode(wav_data).decode("utf-8") #since misty stores base64 string for audio, we use this to convert the wav file to base64 string

        misty.save_audio(fileName='generated_speech.wav', data=base64_encoded, immediatelyApply=False, overwriteExisting=True) #saving the audio to misty
        misty.play_audio('generated_speech.wav', volume=50) #playing audio from misty's internal memory


'''This code is used capture audio using misty's mic and generate transcript using speech interface module. 
From text transcript we generate speech and save audio file onto misty.
Essentially now we make misty repeat whatever we say.'''

if __name__ == '__main__':
    main()

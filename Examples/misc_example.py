from mistyPy.Robot import Robot
from mistyPy.Events import Events
from mistyPy.RobotCommands import RobotCommands

import base64
from pydub import AudioSegment
import speech_recognition as sr
import io

import time

ip='192.168.1.101'

misty = Robot(ip)

def record_audio():

    input('Press enter to start')
    print("Start Recording")
    time.time()
    misty.start_recording_audio('test.wav')
    time.sleep(3)
    print("Done Recording")
    misty.stop_recording_audio()

    audio = misty.get_audio_file('test.wav', True).json()['result']['base64']

    return audio

aud = record_audio()

# Step 1: Decode the Base64 string
aud = base64.b64decode(aud)

# Step 2: Extract audio data
audio = AudioSegment.from_file(io.BytesIO(aud), format='wav')

print('Transcribing text ......')

# Step 3: Convert audio to text
recognizer = sr.Recognizer()
with sr.AudioFile(audio.export("temp.wav", format="wav")) as source:
    audio_data = recognizer.record(source)
    transcribed_text = recognizer.recognize_google(audio_data, language='en')

print(transcribed_text)

# Step 4: Send text to Misty
misty.speak(transcribed_text, 2000, 0.75)
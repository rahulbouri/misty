from mistyPy.Robot import Robot
from mistyPy.Events import Events
from mistyPy.RobotCommands import RobotCommands

import base64
from pydub import AudioSegment
import speech_recognition as sr
import io

ip='192.168.1.101'

misty=Robot(ip)

# audio = misty.get_audio_file('capture_HeyMisty.wav', True).json()['result']['base64'] #capture_HeyMisty.wav

# # Step 1: Decode the Base64 string
# audio = base64.b64decode(audio)

# # Step 2: Extract audio data
# audio = AudioSegment.from_file(io.BytesIO(audio), format='wav')

# print('Transcribing text ......\n')

# # Step 3: Convert audio to text
# recognizer = sr.Recognizer()
# with sr.AudioFile(audio.export("capture_HeyMisty.wav", format="wav")) as source:
#     audio_data = recognizer.record(source)
#     transcribed_text = recognizer.recognize_google(audio_data, language='en')

# print(transcribed_text)

misty.play_audio('capture_HeyMisty.wav', volume=20)
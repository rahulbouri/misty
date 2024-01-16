from mistyPy.Robot import Robot
from mistyPy.Events import Events
from mistyPy.RobotCommands import RobotCommands

import base64
from pydub import AudioSegment
import speech_recognition as sr
import io
import wave

def base64_to_wav(base64_audio_string, output_file_path):
    binary_audio_data = base64.b64decode(base64_audio_string)

    with wave.open(output_file_path, 'wb') as wav_file:
        # Set WAV file parameters based on your audio
        num_channels = 1  # Mono audio
        sample_width = 2  # 2 bytes per sample (16-bit)
        frame_rate = 16000  # Sample rate (e.g., 44.1 kHz)
        num_frames = len(binary_audio_data) // (num_channels * sample_width)
        compression_type = 'NONE'  # No compression

        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(frame_rate)
        wav_file.setnframes(num_frames)
        wav_file.setcomptype(compression_type, 'not compressed')
        wav_file.writeframes(binary_audio_data)

ip='192.168.1.101'

misty=Robot(ip)

audio = misty.get_audio_file('test.wav', True).json()['result']['base64'] #capture_HeyMisty.wav

base64_to_wav(audio, 'temp.wav')

print("Done")

# # Step 1: Decode the Base64 string
# audio = base64.b64decode(audio)

# # Step 2: Extract audio data
# audio = AudioSegment.from_file(io.BytesIO(audio), format='wav')

# print('Transcribing text ......\n')

# # Step 3: Convert audio to text
# recognizer = sr.Recognizer()
# with sr.AudioFile(audio.export("test.wav", format="wav")) as source:
#     audio_data = recognizer.record(source)
#     transcribed_text = recognizer.recognize_google(audio_data, language='en')

# print(transcribed_text)

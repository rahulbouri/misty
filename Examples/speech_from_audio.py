from mistyPy.Robot import Robot
from mistyPy.Events import Events
from mistyPy.RobotCommands import RobotCommands

import base64
from pydub import AudioSegment
import speech_recognition as sr
import io
import time
import soundfile as sf
import torch

from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN



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

tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")


def main():

    audio = record_audio()

    # Step 1: Decode the Base64 string
    audio = base64.b64decode(audio)

    # Step 2: Extract audio data
    audio = AudioSegment.from_file(io.BytesIO(audio), format='wav')

    print('Transcribing text ......')

    # Step 3: Convert audio to text
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio.export("temp.wav", format="wav")) as source:
        audio_data = recognizer.record(source)
        transcribed_text = recognizer.recognize_google(audio_data, language='en')

    print(transcribed_text)

    # Running the TTS
    mel_output, mel_length, alignment = tacotron2.encode_text(transcribed_text)

    # Running Vocoder (spectrogram-to-waveform)
    waveforms = hifi_gan.decode_batch(mel_output)

    waveform_np = waveforms.squeeze().cpu().numpy()
    output_file = "TTS_audio.wav"  # Name of the output WAV file
    sf.write(output_file, waveform_np, samplerate=22050)

    with open(output_file, "rb") as wav_file:
        wav_data = wav_file.read()
        base64_encoded = base64.b64encode(wav_data).decode("utf-8")

    misty.save_audio(fileName='test_file.wav', data=base64_encoded, immediatelyApply=True, overwriteExisting=True)


main()
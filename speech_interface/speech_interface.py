"""
##############################################################################
TCS CONFIDENTIAL
__________________
Copyright : [2021] TATA Consultancy Services Ltd.
All Rights Reserved.

NOTICE:  All information contained herein is, and remains
the property of TATA Consultancy Services Ltd. and its suppliers,
if any.  The intellectual and technical concepts contained
herein are proprietary to TATA Consultancy Services Ltd.
and its suppliers and may be covered by Foreign Patents,
patents in process, and are protected by trade secret or copyright law.
Dissemination of this information or reproduction of this material
is strictly forbidden unless prior written permission is obtained
from TATA Consultancy Services Ltd.
##############################################################################

Author : Chayan Sarkar (sarkar.chayan@tcs.com),
         Pradip Pramanick (pradip.pramanick@tcs.com)

Created : 20 September, 2021

"""
import webrtcvad
import base64
import pyaudio
import wave
from pydub import AudioSegment
import io
import threading
import numpy as np
import queue
import time
import sounddevice as sd

from speech_interface.voice_activity_detection import VoiceActivityDetection
from speech_interface.wakeword_detection.detect_wakeword import WakewordDetection

from mistyPy.Robot import Robot
from mistyPy.Events import Events
from mistyPy.RobotCommands import RobotCommands

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

def read_wave_file(file_path):
    with wave.open(file_path, 'rb') as wf:
        audio_data = wf.readframes(wf.getnframes())
        sample_rate = wf.getframerate()
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
    return audio_data, sample_rate, num_channels, sample_width

def play_audio(audio_data, stream):
    stream.write(audio_data)

class RecordThread(threading.Thread):
    def __init__(self, audio_queue, channels, rate, chunk, ip):
        threading.Thread.__init__(self)
        self.format         = pyaudio.paInt16
        self.channels       = channels
        self.rate           = rate
        self.chunk          = chunk
        self.ip             = ip

        self.audio_queue    = audio_queue
        self.bRecord        = True
        self.misty          = Robot(self.ip)

    def run(self):

        self.misty.start_recording_audio('test.wav')
        audio = (self.misty.get_audio_file('test.wav', True).json()['result']['base64'])
        audio = base64_to_wav(audio, 'test.wav')
        audio_data, sample_rate, num_channels, sample_width = read_wave_file('test.wav')

        p = pyaudio.PyAudio()
        wav_stream = p.open(format=p.get_format_from_width(2),
                channels=1,
                rate=16000,
                output=True)
        play_audio(audio_data, wav_stream)
        wav_stream = p.open(format=self.format,
                                channels=self.channels,
                                rate=self.rate,
                                input=True,
                                frames_per_buffer=self.chunk)
        
        while self.bRecord:
            self.audio_queue.put((time.time(), wav_stream.read(self.chunk)))

        wav_stream.stop_stream()
        wav_stream.close()
        p.terminate()

        print("recording done")

    def stop_record(self):
        self.bRecord = False


class SpeechInterface:
    def __init__(self,ip):

        args = SPArgs()

        # parameters related to sound capture
        self.RATE                   = args.rate                                         # 16 kHz
        self.CHANNELS               = args.channels                                     # 1 (mono)
        self.CHUNK                  = int(args.rate * args.chunk_duration)              # 0.02 s or 20 ms chunk
        self.ip                     = ip

        # parameters related to voice activity
        self.MAX_CHUNKS             = int(self.RATE / self.CHUNK * args.max_duration)       # 10 s recording at max
        self.SILENCE_THRESHOLD      = int(self.RATE / self.CHUNK * args.silence_threshold)  # 2 s silence
        self.HALF_RATE              = int(self.RATE / 2)
        self.QUARTER_RATE           = int(self.RATE / 4)
        self.MISSED_CHUNK_THRESHOLD = 15

        # initiate voice activity detection
        # self.vad = webrtcvad.Vad(3)
        self.vad = VoiceActivityDetection()

        # initiate queue to store recorded audio data
        self.audio_queue = queue.Queue()

        # create the audio recording thread
        self.audio_record = RecordThread(self.audio_queue,
                                         channels=self.CHANNELS,
                                         rate=self.RATE,
                                         chunk=self.CHUNK, ip=self.ip)

        # create the wakeword detection model
        self.wakeword_model = WakewordDetection(wakeword='avixa', mode='single', thresh=0.65)
        # self.wakeword_model = WakewordDetection(wakeword='avixa', mode='max', thresh=0.65)

        # asr callable function
        # self.

        self.misty = Robot(ip)

    def __wakeword_detection(self, audio_frames):
        if self.wakeword_model.detect(audio_frames):
            print("\n*** Wake word detected ***")
            return True
        else:
            return False

    def __passive_listening(self):
        frame = np.array([], dtype="int16")
        is_speech = False
        missed_chunk = 0
        # detection = False

        while True:
            try:
                _, data_bytes = self.audio_queue.get()

                data_array = np.frombuffer(data_bytes, dtype="int16")
                if self.vad.is_speech(data_array, self.RATE):
                    frame = np.concatenate([frame, data_array])
                    missed_chunk = 0
                    if not is_speech:
                        # print("Speech detected  ", type(data_bytes), len(data_bytes))
                        is_speech = True
                    if frame.size > self.RATE:
                        padded_frame = frame.astype(np.float32, order='C') / 32768.0
                        detection = self.__wakeword_detection(padded_frame[:self.RATE])
                        if detection:
                            return
                        frame = frame[self.QUARTER_RATE:]
                else:
                    if frame.size > self.HALF_RATE:
                        if frame.size < self.RATE:
                            padded_frame = np.zeros(self.RATE, dtype=np.float32)
                            padded_frame[:frame.size] = frame.astype(np.float32, order='C') / 32768.0
                        else:
                            padded_frame = frame.astype(np.float32, order='C') / 32768.0
                        t = 0
                        while padded_frame.size >= t + self.RATE:
                            detection = self.__wakeword_detection(padded_frame[t:t+self.RATE])
                            if detection:
                                return
                            else:
                                # print("\n*** Not detected ***")
                                t += self.QUARTER_RATE
                        is_speech = False
                        frame = np.array([], dtype="int16")

                    if is_speech:
                        missed_chunk += 1
                        if missed_chunk >= self.MISSED_CHUNK_THRESHOLD:
                            is_speech = False
                            frame = np.array([], dtype="int16")
                        else:
                            frame = np.concatenate([frame, np.frombuffer(data_bytes, dtype="int16")])

            except KeyboardInterrupt:
                self.misty.stop_recording_audio()
                print("\n*** Stopped Listening ***\n")
                break

    def __record_relevant_audio(self):
        frames = np.array([], dtype="int16")
        missed_chunk = 0
        for i in range(self.MAX_CHUNKS):
            _, data_bytes = self.audio_queue.get()
            data_array = np.frombuffer(data_bytes, dtype="int16")
            frames = np.concatenate([frames, data_array])
            if self.vad.is_speech(data_array, self.RATE):
                missed_chunk = 0
            else:
                missed_chunk += 1
                if missed_chunk > self.SILENCE_THRESHOLD:
                    break

        frames = frames.astype(dtype=np.double) / 32768.0
        self.misty.stop_recording_audio()
        self.audio_record.stop_record()

        return frames

    def start_speech_interface(self):
        # initiate queue to store recorded audio data
        self.audio_queue = queue.Queue()

        # create the audio recording thread
        self.audio_record = RecordThread(self.audio_queue,
                                         channels=self.CHANNELS,
                                         rate=self.RATE,
                                         chunk=self.CHUNK, ip=self.ip)

        self.audio_record.start()

        self.__passive_listening()

        audio_data = self.__record_relevant_audio()

        return audio_data


class SPArgs:

    rate = 16000                            # 16 kHz
    channels = 1                            # 1 (mono)
    chunk_duration = 0.02                   # 20 ms chunk

    max_duration = 10                       # 10 s recording at max
    silence_threshold = 2                   # 2 s silence


def record_audio(record_duration=4, samplerate=16000):

    # record audio for the duration and samplerate
    audio_data = sd.rec(int(record_duration * samplerate), samplerate=samplerate, channels=1)

    # wait until the audio is recored for the duration
    sd.wait()

    # print(type(audio_data), audio_data.shape)
    #
    # # convert the audio data (numpy array) to double
    # audio_data = np.squeeze(audio_data)
    # print(type(audio_data), audio_data.shape, type(audio_data[0]))
    # audio_data = audio_data.astype(dtype=np.double)

    # return the numpy audio data
    return audio_data

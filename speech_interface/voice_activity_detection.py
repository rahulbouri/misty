# inspired from https://maelfabien.github.io/machinelearning/Speech4/#read-the-input-file-and-convert-it-to-mono
import numpy as np


class VoiceActivityDetection:

    def __init__(self):
        self.SPEECH_START_BAND = 300
        self.SPEECH_END_BAND = 3000
        self.THRESHOLD = 0.6

    @staticmethod
    def _calculate_frequencies(audio_data, rate):
        data_freq = np.fft.fftfreq(len(audio_data), 1.0 / rate)
        data_freq = data_freq[1:]
        return data_freq

    @staticmethod
    def _calculate_energy(audio_data):
        data_ampl = np.abs(np.fft.fft(audio_data))
        data_ampl = data_ampl[1:]
        return data_ampl ** 2

    def _connect_energy_with_frequencies(self, data, rate):

        data_freq = self._calculate_frequencies(data, rate)
        data_energy = self._calculate_energy(data)

        energy_freq = {}
        for (i, freq) in enumerate(data_freq):
            if abs(freq) not in energy_freq:
                energy_freq[abs(freq)] = data_energy[i] * 2
        return energy_freq

    def _sum_energy_in_band(self, energy_frequencies):
        sum_energy = 0
        for f in energy_frequencies.keys():
            if self.SPEECH_START_BAND < f < self.SPEECH_END_BAND:
                sum_energy += energy_frequencies[f]
        return sum_energy

    def is_speech(self, audio_data, rate):
        # Full energy
        energy_freq = self._connect_energy_with_frequencies(audio_data, rate)
        sum_full_energy = sum(energy_freq.values())

        # Voice energy
        sum_voice_energy = self._sum_energy_in_band(energy_freq)

        # Speech ratio
        speech_ratio = sum_voice_energy / (sum_full_energy + 0.000001)

        if speech_ratio > self.THRESHOLD:
            return True
        else:
            return False

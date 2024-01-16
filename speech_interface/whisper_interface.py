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
from transformers import AutoProcessor, WhisperForConditionalGeneration


class SpeechInterfaceWhisper:
    def __init__(self, processor="speech_interface/whisper_base",
                 model="speech_interface/whisper_base",
                 sampling_rate=16000):

        self.processor = AutoProcessor.from_pretrained(processor)
        self.model = WhisperForConditionalGeneration.from_pretrained(model)
        self.sampling_rate = sampling_rate

    def transcribe(self, audio_input):
        inputs = self.processor(audio_input, sampling_rate=self.sampling_rate, return_tensors="pt")
        input_features = inputs.input_features

        generated_ids = self.model.generate(inputs=input_features)

        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return transcription

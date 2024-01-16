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

Author : Chayan Sarkar (sarkar.chayan@tcs.com), Pradip Pramanick (pradip.pramanick@tcs.com)
Created : 20 September, 2021

"""

import numpy as np


class SpeechInterface:
    def __init__(self, config):
        self.asr_model = config['ASR_MODEL']
        print("**********", self.asr_model)

        if self.asr_model == 'whisper':
            from speech_interface.whisper_infer import Whisper_infer
            self.asr = Whisper_infer()

        elif self.asr_model == 'BAP_BSD':
            from speech_interface.BAP_BSD_inference_wav2vec2 import BAP_BSD_inference_of_wav2vec2
            tokenizer = config['ASR_TOKENIZER']
            model = config['ASR_MODEL_PATH']
            sampling_rate = config['SAMPLING_RATE']
            lm_path = config['LM_PATH']
            vocab_path = config['BIAS_VOCAB_PATH']
            self.asr = BAP_BSD_inference_of_wav2vec2(tokenizer, model, bias_path=vocab_path, lm_path=lm_path,
                                                     sampling_rate=sampling_rate)
        elif self.asr_model == 'BAP_BSD_CTC_Conformer':
            from speech_interface.BAP_BSD_inference_ctc_conformer import BAP_BSD_inference_of_ctc_conformer
            model = config['ASR_MODEL_PATH']
            if config['USE_LM']:
                lm_path = config['LM_PATH']
            else:
                lm_path = None
            if config['USE_BIAS']:
                bias_vocab = []
                for line in open(config['BIAS_VOCAB_PATH']).readlines():
                    bias_vocab.append(line.strip())
            self.asr = BAP_BSD_inference_of_ctc_conformer('/Users/user/Projects/AssistiveRobo/stt_en_conformer_ctc_small', lm_path, bias_vocab)
            print('\n *****'
                  '\nBAP-BSD Decoder for ctc_conformer initialized with the following options:')
            print('using LM:', bool(lm_path))
            print('using Biasing:', bool(bias_vocab))
            print('*****/n')
        else:
            print("Error! Unknown asr model - ", config['ASR_MODEL'])

    def transcribe(self, audio_data, bias=None):
        # TODO CHANGE DUMMY INPUT
        # text = input("dummy transcription \n ")
        # return text
        if self.asr_model in ['BAP_BSD', 'BAP_BSD_CTC_Conformer']:
            audio_data = np.frombuffer(audio_data, dtype=np.int16)
        return self.asr.transcribe(audio_data, bias=bias)

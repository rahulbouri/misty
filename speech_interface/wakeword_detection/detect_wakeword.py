import torch
import librosa
import json
import numpy as np
from speech_interface.wakeword_detection.utils.misc import get_model
from speech_interface.wakeword_detection.config_parser import get_config


class WakewordDetection:

    def __init__(self, wakeword='avixa',
                 mode='single',
                 win_len=0.8,
                 stride=0.1,
                 thresh=0.65,
                 conf='speech_interface/wakeword_detection/model/base_config.yaml'):
        ######################
        # create model
        ######################
        self.config = get_config(conf)
        self.model = get_model(self.config["hparams"]["model"])

        ######################
        # load weights
        ######################
        ckpt = torch.load(self.config['check_point'], map_location="cpu")
        self.model.load_state_dict(ckpt["model_state_dict"])

        ######################
        # set model for inference
        ######################
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        self.label_map = None
        with open(self.config['label_map'], "r") as f:
            self.label_map = json.load(f)

        self.wakeword = wakeword
        self.win_len = win_len
        self.stride = stride
        self.thresh = thresh
        self.mode = mode

    @staticmethod
    def process_window(x, audio_settings):
        x = librosa.feature.melspectrogram(y=x, **audio_settings)
        x = librosa.feature.mfcc(S=librosa.power_to_db(x), n_mfcc=audio_settings["n_mels"])
        x = torch.from_numpy(x).float().unsqueeze(0)
        x = x.unsqueeze(0)
        return x

    @staticmethod
    def process_window2(x, sr, audio_settings):
        x = librosa.util.fix_length(x, size=sr)
        x = librosa.feature.melspectrogram(y=x, **audio_settings)
        x = librosa.feature.mfcc(S=librosa.power_to_db(x), n_mfcc=audio_settings["n_mels"])
        return x

    @torch.no_grad()
    def get_clip_pred(self, x):
        """Performs clip-level inference."""

        audio_settings = self.config["hparams"]["audio"]
        data = self.process_window(x, audio_settings)
        data = data.to(self.device)

        out = self.model(data)
        out = torch.nn.functional.softmax(out, dim=-1).max(1)
        # locations = [self.id2label[tops.indices[i].item()] for i in range(len(tops.indices))]
        # probs = tops.values.tolist()

        # pred = out.argmax(1).cpu().item()
        pred = out.indices[0].item()
        prob = out.values[0].item()

        return pred, prob

    @torch.no_grad()
    def get_clip_pred2(self, x):
        """Performs clip-level inference."""

        audio_settings = self.config["hparams"]["audio"]
        sr = audio_settings["sr"]
        win_len, stride = int(self.win_len * sr), int(self.stride * sr)

        windows, result = [], []

        slice_positions = np.arange(0, len(x) - win_len + 1, stride)

        for b, i in enumerate(slice_positions):
            windows.append(
                self.process_window2(x[i: i + win_len], sr, audio_settings)
            )

        windows = torch.from_numpy(np.stack(windows)).float().unsqueeze(1)
        windows = windows.to(self.device)
        out = self.model(windows)
        conf, preds = out.softmax(1).max(1)
        conf, preds = conf.cpu().numpy().reshape(-1, 1), preds.cpu().numpy().reshape(-1, 1)

        res = np.hstack([preds, conf])
        res = res[res[:, 1] > self.thresh].tolist()
        if len(res):
            result.extend(res)

        #######################
        # pred aggregation
        #######################
        pred = []
        if len(result):
            result = np.array(result)

            if self.mode == "max":
                pred = result[result[:, 1].argmax()][0]
                if self.label_map is not None:
                    pred = self.label_map[str(int(pred))]
            elif self.mode == "n_voting":
                pred = np.bincount(result[:, 0].astype(int)).argmax()
                if self.label_map is not None:
                    pred = self.label_map[str(int(pred))]
            elif self.mode == "multi":
                if self.label_map is not None:
                    pred = list(map(lambda a: [self.label_map[str(int(a[0]))], a[1], a[2], a[3]], result))
                else:
                    pred = result.tolist()
        return pred

    def detect(self, x):
        if self.mode == 'single':
            pred, prob = self.get_clip_pred(x)
            if prob >= self.thresh and self.label_map[str(pred)] == self.wakeword:
                return True
            else:
                return False
        else:
            pred = self.get_clip_pred2(np.concatenate((x, np.zeros(3200))))
            if pred == self.wakeword:
                return True
            else:
                return False


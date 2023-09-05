# https://github.com/Data-Science-kosta/Speech-Emotion-Classification-with-PyTorch
import numpy as np
import pandas as pd
import librosa
import joblib
import torch
import torch.nn as nn
import pydub
from utils import audiosegment_to_librosawav


DEVICE = torch.device('cpu')

EMOTIONS = {0: 'surprise', 1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear',
            7: 'disgust'}  # surprise je promenjen sa 8 na 0
SAMPLE_RATE = 48000


def get_mel_spectrogram(audio, sample_rate):
    mel_spec = librosa.feature.melspectrogram(y=audio,
                                              sr=sample_rate,
                                              n_fft=1024,
                                              win_length=512,
                                              window='hamming',
                                              hop_length=256,
                                              n_mels=128,
                                              fmax=sample_rate / 2
                                              )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


class ParallelModel(nn.Module):
    def __init__(self, num_emotions):
        super().__init__()
        # conv block
        self.conv2Dblock = nn.Sequential(
            # 1. conv block
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
            # 2. conv block
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
            # 3. conv block
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
            # 4. conv block
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3)
        )
        # Transformer block
        self.transf_maxpool = nn.MaxPool2d(kernel_size=[2, 4], stride=[2, 4])
        transf_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=512, dropout=0.4,
                                                  activation='relu')
        self.transf_encoder = nn.TransformerEncoder(transf_layer, num_layers=4)
        # Linear softmax layer
        self.out_linear = nn.Linear(320, num_emotions)
        self.dropout_linear = nn.Dropout(p=0)
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # conv embedding
        conv_embedding = self.conv2Dblock(x)  # (b,channel,freq,time)
        conv_embedding = torch.flatten(conv_embedding, start_dim=1)  # do not flatten batch dimension
        # transformer embedding
        x_reduced = self.transf_maxpool(x)
        x_reduced = torch.squeeze(x_reduced, 1)
        x_reduced = x_reduced.permute(2, 0, 1)  # requires shape = (time,batch,embedding)
        transf_out = self.transf_encoder(x_reduced)
        transf_embedding = torch.mean(transf_out, dim=0)
        # concatenate
        complete_embedding = torch.cat([conv_embedding, transf_embedding], dim=1)
        # final Linear
        output_logits = self.out_linear(complete_embedding)
        output_logits = self.dropout_linear(output_logits)
        output_softmax = self.out_softmax(output_logits)
        return output_logits, output_softmax


class SpeechEmotionRecognition:
    def __init__(self):
        self.model = ParallelModel(len(EMOTIONS))
        self.model.load_state_dict(torch.load('models/ser/cnn_transf_parallel_model.pt', map_location=DEVICE))
        self.scaler = joblib.load('models/ser/scaler.gz')

    def _predict_batch(self, X):
        self.model.eval()
        with torch.no_grad():
            _, output_softmax = self.model(X)
            # predictions = torch.argmax(output_softmax, dim=1)
            proba = output_softmax.numpy()
            predictions = pd.DataFrame(proba, columns=list(EMOTIONS.values())).to_dict('records')
        return predictions

    def inference(self, sound_window_buffer: pydub.AudioSegment):

        audio = audiosegment_to_librosawav(sound_window_buffer)
        signal = np.zeros((int(SAMPLE_RATE * 3, )))
        signal[:len(audio)] = audio
        x = get_mel_spectrogram(signal, SAMPLE_RATE)

        # add channel dim
        x = np.expand_dims(x, 0)

        c, h, w = x.shape
        x = self.scaler.transform(x.reshape(c, -1))
        x = x.reshape(c, h, w)

        X = np.stack([x], axis=0)
        X = torch.tensor(X, device=DEVICE).float()

        return self._predict_batch(X)[0]

# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/9/22
"""
demo_cli
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

import json
import numpy as np
import yaml
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from .model import Tacotron2, load_model
from .hparams import create_hparams, Dict2Obj
# from .data_utils import transform_mel, transform_text, transform_f0, transform_embed, transform_speaker
from .layers import TacotronSTFT
from .data_utils import TextMelLoader, TextMelCollate
# from .plotting_utils import plot_mel_alignment_gate_audio

_model = None
_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_mellotron_model(model_path='', hparams_path='', device=None):
    """
    导入训练得到的checkpoint模型。
    """
    global _model
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if str(hparams_path).endswith('.yml'):
        with open(hparams_path, "r") as f:
            hparams = yaml.load(f, Loader=yaml.FullLoader)
    else:
        hparams = json.load(open(hparams_path, encoding='utf8'))
    hparams = Dict2Obj(hparams)
    _model = load_model(hparams).to(device).eval()
    _model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])


def load_mellotron_torch(model_path='', device=None):
    """
    用torch.load直接导入模型文件，不需要导入模型代码。
    """
    global _model
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _model = torch.load(model_path, map_location=device)
    return _model


def is_loaded():
    """
    判断模型是否已经loaded。
    """
    global _model
    return _model is not None


def generate_mel(text, style, speaker, f0, **kwargs):
    """
    用语音合成模型把文本转为mel频谱。
    """
    global _model
    if not is_loaded() and 'model_path' in kwargs:
        load_mellotron_torch(model_path=kwargs['model_path'])

    with torch.no_grad():
        mels, mels_postnet, gates, alignments = _model.inference((text, style, speaker, f0))
        gates = torch.sigmoid(gates)
        alignments = alignments.permute(0, 2, 1)
        return mels, mels_postnet, gates, alignments


def generate_mel_batch(model, inpath, batch_size, hparams, **kwargs):
    """Handles all the validation scoring and printing"""
    # global _model
    # if not is_loaded():
    #     load_mellotron_torch(**kwargs)

    model.eval()
    with torch.no_grad():
        valset = TextMelLoader(inpath, hparams, speaker_ids={}, mode=hparams.train_mode)
        collate_fn = TextMelCollate(n_frames_per_step=hparams.n_frames_per_step, mode='val')
        # fixme num_workers>=1则EOFError: Ran out of input，mp.spawn的报错。
        val_loader = DataLoader(valset, sampler=None, num_workers=0,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, drop_last=False, collate_fn=collate_fn)
        mels, mels_postnet, gates, alignments = [], [], [], []
        for i, batch in enumerate(tqdm(val_loader, 'mellotron', ncols=100)):
            x, y = model.parse_batch(batch)  # y: 2部分
            y_pred = model.inference((x[0], x[2], x[5], x[6]))

            mel_outputs, mel_outputs_postnet, gate_outputs, alignment_outputs = y_pred

            # wav_outputs = valset.stft.griffin_lim(mel_outputs_postnet, n_iters=5, denoiser_strength=0)

            out_mels = mel_outputs.data.cpu().numpy()
            out_mels_postnet = mel_outputs_postnet.data.cpu().numpy()
            out_aligns = alignment_outputs.data.cpu().numpy()
            out_gates = torch.sigmoid(gate_outputs.data).cpu().numpy()
            # out_wavs = wav_outputs.data.cpu().numpy()

            for out_mel, out_mel_postnet, out_align, out_gate in zip(out_mels, out_mels_postnet, out_aligns, out_gates):
                end_idx = np.argmax(out_gate > 0.5) or out_gate.shape[0]

                out_mel = out_mel[:, :end_idx]
                out_mel_postnet = out_mel_postnet[:, :end_idx]
                out_align = out_align.T[:, :end_idx]
                out_gate = out_gate[:end_idx]
                # out_wav = out_wav[:end_idx * hparams.hop_length]

                mels.append(out_mel)
                mels_postnet.append(out_mel_postnet)
                gates.append(out_gate)
                alignments.append(out_align)
        return mels, mels_postnet, gates, alignments


class MellotronSynthesizer():
    def __init__(self, model_path, speakers_path, hparams_path, texts_path, device=_device):
        self.device = device

        args_hparams = open(hparams_path, encoding='utf8').read()
        self.hparams = create_hparams(args_hparams)

        self.model = load_model(self.hparams).to(self.device).eval()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device)['state_dict'])

        self.speakers = json.load(open(speakers_path, encoding='utf8'))
        self.texts = [w.strip() for w in open(texts_path, encoding='utf8')]
        self.texts_path = texts_path
        self.stft = TacotronSTFT(
            self.hparams.filter_length, self.hparams.hop_length, self.hparams.win_length,
            self.hparams.n_mel_channels, self.hparams.sampling_rate, self.hparams.mel_fmin,
            self.hparams.mel_fmax)

        self.dataloader = TextMelLoader(audiopaths_and_text=texts_path, hparams=self.hparams, speaker_ids=self.speakers)
        self.datacollate = TextMelCollate(1)

    def synthesize_batch(self):
        return generate_mel_batch(self.model, self.texts_path, 4, self.hparams)

    def synthesize(self, text, speaker, audio, with_show=False):
        text_data, mel_data, speaker_data, f0_data = self.dataloader.get_data_train_v2([audio, text, speaker])
        text_encoded = text_data[None, :].long().to(self.device)
        style_input = 0
        speaker_id = speaker_data.to(self.device)
        pitch_contour = f0_data

        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = self.model.inference(
                (text_encoded, style_input, speaker_id, pitch_contour))

        out_mel = mel_outputs_postnet.data.cpu().numpy()[0]
        out_align = alignments.data.cpu().numpy()[0]
        out_gate = torch.sigmoid(gate_outputs.data).cpu().numpy()[0]

        end_idx = np.argmax(out_gate > 0.2) or out_gate.shape[0]

        out_mel = out_mel[:, :end_idx]
        out_align = out_align.T[:, :end_idx]
        out_gate = out_gate[:end_idx]
        return (out_mel, out_align, out_gate) if with_show else out_mel

    def generate_alignment(self, text, speaker, audio, with_show=False):
        text_data, mel_data, speaker_data, f0_data = self.dataloader.get_data_train_v2([audio, text, speaker])
        x, y = self.model.parse_batch(self.datacollate([(text_data, mel_data, speaker_data, f0_data)]))

        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = self.model.forward(x)
            alignments = alignments.permute(1, 0, 2)

        out_mel = mel_outputs_postnet.data.cpu().numpy()[0]
        out_align = alignments.data.cpu().numpy()[0]
        out_gate = torch.sigmoid(gate_outputs.data).cpu().numpy()[0]

        end_idx = np.argmax(out_gate > 0.2) or out_gate.shape[0]

        out_mel = out_mel[:, :end_idx]
        out_align = out_align.T[:, :end_idx]
        out_gate = out_gate[:end_idx]
        return (out_mel, out_align, out_gate) if with_show else out_mel


def save_model(model: MellotronSynthesizer, outpath=''):
    torch.save(model.model, outpath)

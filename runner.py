#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2021-12-16
"""
python ttskit\melgan\trainer.py -c ttskit\melgan\config\default.yaml -n workplace\melgan-samples -p workplace\melgan.kuangdd.chk.pt

"""
import sys
import os
import re
import json
import collections as clt
import shutil
import pathlib
import functools
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import tqdm


def mel():
    from ttskit.mellotron.layers import TacotronSTFT
    M = TacotronSTFT()
    import librosa
    male_path = r'E:\github\ttskit\workplace\kdd-4387.wav'
    female_path = r'E:\github\ttskit\workplace\samples\biaobei\wav\000001.wav'
    wav, sr = librosa.load(male_path, sr=None)
    print(sr)
    import torch
    wav_t = torch.from_numpy(wav)
    import aukit
    from aukit.audio_tuner import tune_speed_pydub
    # mel_t = M.mel_spectrogram(wav_t.unsqueeze(0))
    import matplotlib.pyplot as plt
    # plt.pcolor(mel_t[0].cpu().numpy())
    wav_44k = librosa.resample(wav, sr, 44100)

    wav_22k = librosa.resample(wav, sr, 11025)
    wav_22k = aukit.tune_speed(wav_22k, sr=11050, rate=0.5)

    from aukit.audio_tuner import tune_pitch, tune_pitch_librosa
    # wav_22k_female = aukit.tune_pitch(wav_22k, sr=22050, rate=1.5)
    # plt.plot(wav_22k)
    # plt.show()
    wav_22k = np.clip(wav_22k, a_max=1, a_min=-1)
    wav_22k_t = torch.from_numpy(wav_22k)

    mel_t = M.mel_spectrogram(wav_22k_t.unsqueeze(0))
    plt.pcolor(mel_t[0].cpu().numpy())
    plt.show()

    import aukit
    aukit.play_sound(wav_22k, sr=22050)
    wav_new = librosa.resample(wav_22k, 11025, 22050)
    # wav_new = aukit.tune_pitch(wav_22k_female, sr=22050, rate=1/1.5)
    # wav_new = aukit.tune_speed(wav_new, sr=22050, rate=2)
    aukit.play_sound(wav_new, sr=22050)
    aukit.play_sound(wav, sr=sr)


def set_audio_format(inpath, outpath):
    """设置音频格式。"""
    import pydub

    aud = pydub.AudioSegment.from_file(inpath)  # 导入音频文件

    aud = aud.set_channels(2)  # 设置通道数
    aud = aud.set_frame_rate(16000)  # 设置采样率
    aud = aud.set_sample_width(3)  # 设置位长

    aud.export(outpath, format='wav')  # 设置文件格式


def run_single(args):
    """做成一个参数的形式。"""
    return set_audio_format(args[0], args[1])


def run_batch(n_workers=10):
    """多进程跑任务。"""
    import multiprocessing as mp

    indir = r'E:\github\ttskit\workplace\samples\biaobei\wav'
    outdir = indir + '-out'
    os.makedirs(outdir, exist_ok=True)

    # 生成任务需要的参数列表
    args_lst = [(os.path.join(indir, w), os.path.join(outdir, w)) for w in os.listdir(indir)]

    pool = mp.Pool(n_workers)  # 构建进程池
    pool.map(run_single, args_lst)  # 多进程跑任务
    pool.close()
    pool.join()


if __name__ == "__main__":
    print(__file__)
    # set_audio_format()

    run_batch()

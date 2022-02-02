import random
import subprocess
import numpy as np
from scipy.io.wavfile import read
import librosa


def get_commit_hash():
    message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return message.strip().decode('utf-8')


def read_wav_np(path):
    sr, wav = read(path)

    if len(wav.shape) == 2:
        wav = wav[:, 0]

    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0

    wav = wav.astype(np.float32)

    return sr, wav


def load_wav(fpath, sr_force=None):
    wav, sr = librosa.load(fpath, sr=None)
    if (sr_force is not None) and (sr != sr_force):
        wav = librosa.resample(wav, orig_sr=sr, target_sr=sr_force)

    # fixme 标准化，音量一致
    wav = 0.9 * wav / max(np.max(np.abs(wav)), 0.01)
    # out = np.clip(wav, -1, 1) * (2 ** 15 - 1)
    # out = out.astype(int)
    return (sr_force or sr), wav

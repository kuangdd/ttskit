import os
import glob
import tqdm
import torch
import argparse
from scipy.io.wavfile import write

from .model.generator import Generator
from .utils.hparams import HParam, load_hparam_str

MAX_WAV_VALUE = 32768.0

_model = None


def load_melgan_model(model_path='', device=None):
    """
    导入训练模型得到的checkpoint模型文件。
    """
    global _model
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if _model is None:
        checkpoint = torch.load(model_path)
        if args.config is not None:
            hp = HParam(args.config)
        else:
            hp = load_hparam_str(checkpoint['hp_str'])

        _model = Generator(hp.audio.n_mel_channels).to(device)
        _model.load_state_dict(checkpoint['model_g'])
        _model.eval(inference=False)


def load_melgan_torch(model_path='', device=None):
    """
    用torch.load直接导入模型文件，不需要导入模型代码。
    """
    global _model
    global _denoiser
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _model = torch.load(model_path, map_location=device)
    return _model


def is_loaded():
    """
    判断模型是否已经导入。
    """
    global _model
    return _model is not None


def generate_wave(mel, **kwargs):
    """
    用声码器模型把mel频谱转为音频信号。
    """
    global _model
    if not is_loaded() and 'model_path' in kwargs:
        load_melgan_torch(kwargs['model_path'])
    with torch.no_grad():
        wav = _model.inference(mel)
        return wav.detach() / MAX_WAV_VALUE


def main(args):
    checkpoint = torch.load(args.checkpoint_path)
    if args.config is not None:
        hp = HParam(args.config)
    else:
        hp = load_hparam_str(checkpoint['hp_str'])

    model = Generator(hp.audio.n_mel_channels).cuda()
    model.load_state_dict(checkpoint['model_g'])
    model.eval(inference=False)

    with torch.no_grad():
        for melpath in tqdm.tqdm(glob.glob(os.path.join(args.input_folder, '*.mel'))):
            mel = torch.load(melpath)
            if len(mel.shape) == 2:
                mel = mel.unsqueeze(0)
            mel = mel.cuda()

            audio = model.inference(mel)
            audio = audio.cpu().detach().numpy()

            out_path = melpath.replace('.mel', '_reconstructed_epoch%04d.wav' % checkpoint['epoch'])
            write(out_path, hp.audio.sampling_rate, audio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None,
                        help="yaml file for config. will use hp_str from checkpoint if not given.")
    parser.add_argument('-p', '--checkpoint_path', type=str, required=True,
                        help="path of checkpoint pt file for evaluation")
    parser.add_argument('-i', '--input_folder', type=str, required=True,
                        help="directory of mel-spectrograms to invert into raw audio. ")
    args = parser.parse_args()

    main(args)

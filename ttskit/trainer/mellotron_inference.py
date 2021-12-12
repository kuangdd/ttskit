#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/11/28
"""
mellotron_inference
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--checkpoint_path', type=str,
                        default=r"workplace/mellotron-mspk-samples/checkpoint/mellotron-000000.pt",
                        help='模型路径。')
    parser.add_argument('--is_simple', type=int, default=1,
                        help='是否简易模式。')
    parser.add_argument('-s', '--speaker_path', type=str,
                        default=r"workplace/mellotron-mspk-samples/metadata/speakers.json",
                        help='发音人映射表路径。')
    parser.add_argument('-a', '--audio_path', type=str,
                        default=r"../data/samples/wav",
                        help='参考音频路径。')
    parser.add_argument('-t', '--text_path', type=str,
                        default=r"workplace/mellotron-mspk-samples/metadata/validation.txt",
                        help='文本路径。')
    parser.add_argument("-o", "--out_dir", type=Path, default=r"workplace/mellotron-mspk-samples/test/mellotron-000000",
                        help='保存合成的数据路径。')
    parser.add_argument("-p", "--play", type=int, default=0,
                        help='是否合成语音后自动播放语音。')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--hparams_path', type=str,
                        default=r"workplace/mellotron-mspk-samples/metadata/hparams.json",
                        required=False, help='comma separated name=value pairs')
    parser.add_argument("-e", "--encoder_model_fpath", type=Path,
                        default=r"../models/encoder/saved_models/ge2e_pretrained.pt",
                        help="Path your trained encoder model.")
    parser.add_argument("--save_model_path", type=str,
                        default=r"workplace/mellotron-mspk-samples/mellotron-000000.samples.pt",
                        help='保存模型为可以直接torch.load的格式')
    parser.add_argument("--cuda", type=str, default='0',
                        help='设置CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    return args


args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

import matplotlib.pyplot as plt
import aukit
import time
import json
import traceback
import torch
import numpy as np
import shutil
import re
import yaml
import unidecode
from tqdm import tqdm

from ttskit.mellotron.inference import MellotronSynthesizer
from ttskit.mellotron.inference import save_model

# from ttskit.utils.texthelper import xinqing_texts
# from ttskit.utils.argutils import locals2dict

filename_formatter_re = re.compile(r'[\s\\/:*?"<>|\']+')


def convert_input(text):
    fpath = Path(text)
    if fpath.is_file():
        text_inputs = [w.strip() for w in open(fpath, encoding='utf8')]
    elif fpath.is_dir():
        text_inputs = [str(w) for w in sorted(fpath.glob('**/*')) if w.is_file()]
    else:
        text_inputs = [text]
    return text_inputs


def plot_mel_alignment_gate_audio(mel, alignment, gate, audio, figsize=(16, 16)):
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    axes = axes.flatten()
    axes[0].imshow(mel, aspect='auto', origin='bottom', interpolation='none')
    axes[1].imshow(alignment, aspect='auto', origin='bottom', interpolation='none')
    axes[2].scatter(range(len(gate)), gate, alpha=0.5, color='red', marker='.', s=1)
    axes[2].set_xlim(0, len(gate))
    axes[3].scatter(range(len(audio)), audio, alpha=0.5, color='blue', marker='.', s=1)
    axes[3].set_xlim(0, len(audio))

    axes[0].set_title("mel")
    axes[1].set_title("alignment")
    axes[2].set_title("gate")
    axes[3].set_title("audio")

    plt.tight_layout()


def locals2dict(locals_dict):
    obj = {k: (str(v) if isinstance(v, Path) else v)
           for k, v in locals_dict.items() if isinstance(v, (int, float, str, Path, bool))}
    return obj


if __name__ == "__main__":
    # args_hparams = open(args.hparams_path, encoding='utf8').read()
    # _hparams = create_hparams(args_hparams)
    #
    # model_path = args.checkpoint_path
    # load_model_mellotron(model_path, hparams=_hparams)

    ## Print some environment information (for debugging purposes)
    print("Running a test of your configuration...\n")
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
              "%.1fGb total memory.\n" %
              (torch.cuda.device_count(),
               device_id,
               gpu_properties.name,
               gpu_properties.major,
               gpu_properties.minor,
               gpu_properties.total_memory / 1e9))

    if args.is_simple:
        workdir = Path(args.checkpoint_path).parent.parent
        model_stem = Path(args.checkpoint_path).stem
        model_path = args.checkpoint_path
        speakers_path = workdir.joinpath('metadata', 'speakers.json')
        hparams_path = workdir.joinpath('metadata', 'hparams.json')
        texts_path = workdir.joinpath('metadata', 'validation.txt')
        audio_path = workdir.joinpath('metadata', 'validation.txt')
        save_model_path = workdir.joinpath(f'{model_stem}.{workdir.stem}.pt')
        out_dir = workdir.joinpath('test', model_stem)
    else:
        model_path = args.checkpoint_path
        speakers_path = args.speaker_path
        hparams_path = args.hparams_path
        texts_path = args.text_path
        audio_path = args.audio_path
        save_model_path = args.save_model_path
        out_dir = args.out_dir

    msyner = MellotronSynthesizer(model_path=model_path,
                                  speakers_path=speakers_path,
                                  hparams_path=hparams_path,
                                  texts_path=texts_path)

    if save_model_path:
        save_model(msyner, save_model_path)

    speaker_index_dict = json.load(open(speakers_path, encoding='utf8'))
    speaker_name_list = list(speaker_index_dict.keys())

    if args.is_simple:
        example_audio_list = [w.split('\t')[0] for w in convert_input(audio_path)]
        example_text_list = [w.split('\t')[1] for w in convert_input(audio_path)]
        example_speaker_list = [w.split('\t')[2] for w in convert_input(audio_path)]
    else:
        example_audio_list = [w.split('\t')[0] for w in convert_input(audio_path)]
        example_text_list = ['你好，欢迎使用语言合成服务。']
        example_speaker_list = speaker_name_list

    example_audio_list = np.random.choice(example_audio_list, 10)
    example_text_list = np.random.choice(example_text_list, 10)
    example_speaker_list = np.random.choice(example_speaker_list, 10)

    spec = msyner.synthesize(text='你好，欢迎使用语言合成服务。', speaker=speaker_name_list[0], audio=example_audio_list[0])

    ## Run a test

    print("Spectrogram shape: {}".format(spec.shape))
    # print("Alignment shape: {}".format(align.shape))
    wav_inputs = msyner.stft.griffin_lim(torch.from_numpy(spec[None]))
    wav = wav_inputs[0]  # .cpu().numpy()
    print("Waveform shape: {}".format(wav.shape))

    print("All test passed! You can now synthesize speech.\n\n")

    print("Interactive generation loop")
    num_generated = 0
    out_dir.mkdir(exist_ok=True, parents=True)

    # while True:
    #     try:
    #         speaker = input("Speaker:\n")
    #         if not speaker.strip():
    #             speaker = np.random.choice(speaker_names)
    #         print('Speaker: {}'.format(speaker))
    #
    #         text = input("Text:\n")
    #         if not text.strip():
    #             text = np.random.choice(example_texts)
    #         print('Text: {}'.format(text))
    #
    #         audio = input("Audio:\n")
    #         if not audio.strip():
    #             audio = np.random.choice(example_fpaths)
    #         print('Audio: {}'.format(audio))
    # fixme batch的形式合成
    mels, mels_postnet, gates, alignments = msyner.synthesize_batch()
    texts = msyner.texts
    for num in tqdm(list(range(len(mels_postnet))), 'griffinlim', ncols=100):
        # for num, (text, speaker, audio) in enumerate(tqdm(zip(example_text_list, example_speaker_list, example_audio_list),
        #                                                   'mellotron-inference', ncols=100)):
        try:
            spec, align, gate = mels_postnet[num], alignments[num], gates[num]
            audio, text, speaker = texts[num].split('\t')
            # print(f'number: {num}')
            # print(f'text: {text}')
            # print(f'speaker: {speaker}')
            # print(f'audio： {audio}')
            # The synthesizer works in batch, so you need to put your data in a list or numpy array
            # text_input, speaker_input, audio_input = str(text), str(speaker), str(audio)  # 为了能够放到locals
            # print("Creating the spectrogram ...")
            # spec, align, gate = msyner.synthesize(text=text, speaker=speaker, audio=audio, with_show=True)

            # print("Spectrogram shape: {}".format(spec.shape))
            # print("Alignment shape: {}".format(align.shape))

            ## Generating the waveform
            # print("Synthesizing the waveform ...")
            # todo 默认用griffinlim声码器转为声音波形，能测评合成频谱效果，但音质较差，改为melgan声码器。
            wav_outputs = msyner.stft.griffin_lim(torch.from_numpy(spec[None]), n_iters=5)
            wav_output = wav_outputs[0]#.cpu().numpy()

            # print("Waveform shape: {}".format(wav.shape))

            # Save it on the disk
            cur_text = filename_formatter_re.sub('', unidecode.unidecode(text))[:15]
            cur_time = time.strftime('%Y%m%d-%H%M%S')
            out_path = out_dir.joinpath("demo_{}_{}_out.wav".format(cur_time, cur_text))
            aukit.save_wav(wav_output, out_path, sr=msyner.stft.sampling_rate)  # save

            if isinstance(audio, (Path, str)) and Path(audio).is_file():
                ref_path = out_dir.joinpath("demo_{}_{}_ref.wav".format(cur_time, cur_text))
                shutil.copyfile(audio, ref_path)

            fig_path = out_dir.joinpath("demo_{}_{}_fig.jpg".format(cur_time, cur_text))
            plot_mel_alignment_gate_audio(spec, align, gate, wav[::msyner.stft.sampling_rate // 1000])
            plt.savefig(fig_path)
            plt.close()

            yml_path = out_dir.joinpath("demo_{}_{}_info.yml".format(cur_time, cur_text))
            info_dict = locals2dict(locals())
            with open(yml_path, 'wt', encoding='utf8') as fout:
                yaml.dump(info_dict, fout, default_flow_style=False, encoding='utf-8', allow_unicode=True)

            txt_path = out_dir.joinpath("info_dict.txt".format(cur_time))
            with open(txt_path, 'at', encoding='utf8') as fout:
                fout.write('{}\n'.format(json.dumps(info_dict, ensure_ascii=False)))

            num_generated += 1
            # print("\nSaved output as %s\n\n" % out_path)
            if args.play:
                aukit.play_audio(out_path, sr=msyner.stft.sampling_rate)
        except Exception as e:
            print("Caught exception: %s" % repr(e))
            print("Restarting\n")
            traceback.print_exc()

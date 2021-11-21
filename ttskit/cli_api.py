# author: kuangdd
# date: 2021/4/23
"""
### cli_api
语音合成命令行接口。
用命令行调用语音合成。

+ 简单使用
```python
from ttskit import cli_api

args = cli_api.parse_args()
cli_api.tts_cli(args)
# 命令行交互模式使用语音合成。
```

+ 命令行
```
tkcli

usage: tkcli [-h] [-i INTERACTION] [-t TEXT] [-s SPEAKER] [-a AUDIO]
             [-o OUTPUT] [-m MELLOTRON_PATH] [-w WAVEGLOW_PATH] [-g GE2E_PATH]
             [--mellotron_hparams_path MELLOTRON_HPARAMS_PATH]
             [--waveglow_kwargs_json WAVEGLOW_KWARGS_JSON]

语音合成命令行。

optional arguments:
  -h, --help            show this help message and exit
  -i INTERACTION, --interaction INTERACTION
                        是否交互，如果1则交互，如果0则不交互。交互模式下：如果不输入文本或发音人，则为随机。如果输入文本为exit
                        ，则退出。
  -t TEXT, --text TEXT  Input text content
  -s SPEAKER, --speaker SPEAKER
                        Input speaker name
  -a AUDIO, --audio AUDIO
                        Input audio path or audio index
  -o OUTPUT, --output OUTPUT
                        Output audio path. 如果play开头，则播放合成语音；如果.wav结尾，则保存语音。
  -m MELLOTRON_PATH, --mellotron_path MELLOTRON_PATH
                        Mellotron model file path
  -w WAVEGLOW_PATH, --waveglow_path WAVEGLOW_PATH
                        WaveGlow model file path
  -g GE2E_PATH, --ge2e_path GE2E_PATH
                        Ge2e model file path
  --mellotron_hparams_path MELLOTRON_HPARAMS_PATH
                        Mellotron hparams json file path
  --waveglow_kwargs_json WAVEGLOW_KWARGS_JSON
                        Waveglow kwargs json
```

"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

import sys
import os

sys.path.append(os.path.abspath('..'))

import argparse
import json
import numpy as np

from . import sdk_api

mellotron_path = sdk_api._mellotron_path
waveglow_path = sdk_api._waveglow_path
ge2e_path = sdk_api._ge2e_path
mellotron_hparams_path = sdk_api._mellotron_hparams_path

_example_text_list = """去商店买东西一算账1001块，小王对老板说：“一块钱算了。” 
老板说好的。于是小王放下一块钱就走了，
老板死命追了小王五条街又要小王付了1000，
小王感慨：#自然语言理解太难了#
“碳碳键键能能否否定定律一”
书《无线电法国别研究》
要去见投资人，出门时，发现车钥匙下面压了一张员工的小字条，写着“老板，加油！”，
瞬间感觉好有温度，当时心里就泪奔了。
心里默默发誓：我一定会努力的！ 车开了15分钟后，没油了。
他快抱不起儿子了，因为他太胖了""".split('\n')

_example_audio_list = [f'{num}' for num, audio in enumerate(sdk_api._reference_audio_list, 1)]


def parse_args():
    """命令行参数设置。"""
    parser = argparse.ArgumentParser(description='语音合成命令行。')
    parser.add_argument('-i', '--interaction', type=int, default=1,
                        help='是否交互，如果1则交互，如果0则不交互。交互模式下：如果不输入文本或发音人，则为随机。如果输入文本为exit，则退出。')
    parser.add_argument('-t', '--text', type=str, default="欢迎使用语音合成接口。",
                        help='Input text content')
    parser.add_argument('-s', '--speaker', type=str, default="biaobei",
                        help='Input speaker name')
    parser.add_argument('-a', '--audio', type=str, default="0",
                        help='Input audio path or audio index')
    parser.add_argument('-o', '--output', type=str, default='play',
                        help='Output audio path. 如果play开头，则播放合成语音；如果.wav结尾，则保存语音。')
    parser.add_argument('-m', '--mellotron_path', type=str, default=mellotron_path,
                        help='Mellotron model file path')
    parser.add_argument('-w', '--waveglow_path', type=str, default=waveglow_path,
                        help='WaveGlow model file path')
    parser.add_argument('-g', '--ge2e_path', type=str, default=ge2e_path,
                        help='Ge2e model file path')
    parser.add_argument('--mellotron_hparams_path', type=str, default=mellotron_hparams_path,
                        help='Mellotron hparams json file path')
    parser.add_argument('--waveglow_kwargs_json', type=str, default=r'{"denoiser_strength":1,"sigma":1}',
                        help='Waveglow kwargs json')

    args = parser.parse_args()
    return args


def tts_cli(args=None):
    """命令行执行接口。"""
    if args is None:
        args = parse_args()

    sdk_api.load_models(mellotron_path=args.mellotron_path,
                        waveglow_path=args.waveglow_path,
                        ge2e_path=args.ge2e_path,
                        mellotron_hparams_path=args.mellotron_hparams_path)
    waveglow_kwargs = json.loads(args.waveglow_kwargs_json)
    if args.interaction:
        while 1:
            text_input = input('Input text (输入文本或exit退出，不输入则随机):\n')
            if text_input.startswith('exit'):
                return
            json_input = input('Input kwargs (输入控制参数，格式：audio=1,speaker=biaobei，不输入则默认)\n')
            text = text_input or np.random.choice(_example_text_list)
            try:
                kw = dict(w.split('=') for w in json_input.split(','))
            except Exception as e:
                print(e)
                kw = {}
            if not kw.get('audio'):
                if not kw.get('speaker'):
                    audio = np.random.choice(_example_audio_list)
                    speaker = 'tmp'
                else:
                    audio = 'tmp'
                    speaker = kw.get('speaker')
            else:
                audio = kw.get('audio')
                speaker = 'tmp'
            kw = {**kw, **{'audio': audio, 'speaker': speaker}}
            print(f'Text: {text}')
            print(f'Kwargs: {kw}')
            print('TTS running ...')
            kw.pop('audio')
            kw.pop('speaker')
            sdk_api.tts_sdk(text=text,
                            speaker=speaker,
                            audio=audio,
                            output=kw.get('output', args.output),
                            **kw)
    else:
        wav_output = sdk_api.tts_sdk(text=args.text,
                                     speaker=args.speaker,
                                     audio=args.audio,
                                     output=args.output,
                                     **waveglow_kwargs)
    return wav_output


if __name__ == "__main__":
    logger.info(__file__)
    sdk_api.download_resource()
    args = parse_args()
    tts_cli(args)

# -*- coding:utf-8 -*-
# @project: GPT2-NewsTitle
# @filename: http_server.py
# @author: 刘聪NLP
# @contact: logcongcong@gmail.com
# @time: 2020/12/19 20:49
"""
### http_server
语音合成简易界面。
构建简单的语音合成网页服务。

+ 简单使用
```python
from ttskit import http_server

http_server.start_sever()
# 打开网页：http://localhost:9000/ttskit
```

+ 命令行
```
tkhttp

usage: tkhttp [-h] [--device DEVICE] [--host HOST] [--port PORT]

optional arguments:
  -h, --help       show this help message and exit
  --device DEVICE  设置预测时使用的显卡,使用CPU设置成-1即可
  --host HOST      IP地址
  --port PORT      端口号
```

+ 网页界面
![index](ttskit/templates/index.png "index")
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

import sys
import os

sys.path.append(os.path.abspath('..'))

from flask import Flask, request, render_template, Response
import argparse
from gevent import pywsgi as wsgi

import yaml

import ttskit.sdk_api as sdk_api


def set_args():
    """设置所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='设置预测时使用的显卡,使用CPU设置成-1即可')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='IP地址')
    parser.add_argument('--port', type=int, default=9000, help='端口号')
    return parser.parse_args()


def start_sever():
    """部署网页服务。"""
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICE"] = args.device
    app = Flask(__name__)

    @app.route('/')
    def index():
        return 'hello'  # "这是语言合成工具箱网页服务"

    @app.route('/ttskit', methods=['GET', 'POST'])
    def response_request():
        if request.method == 'POST':
            content = request.form.get('content')
            title = request.form.get('title')
            return render_template("index.html")
        content = '欢迎使用语音合成工具箱，请输入需要合成的文本。'
        title = 'format: yaml\nmode: mspk\naudio: 14\nspeaker: Aiyue\nvocoder: melgan\nsigma: 1.0\ndenoiser_strength: 1.2\ngriffinlim_iters: 30\n'
        return render_template("index.html", content=content, title=title)

    @app.route('/synthesize', methods=['GET', 'POST'])
    def synthesize():
        if request.method == 'GET':
            text = request.args.get('text')
            kwargs_str = request.args.get('kwargs')
            kwargs = yaml.load(kwargs_str)
            wav = sdk_api.tts_sdk(text=text, **kwargs)
            return Response(wav, mimetype='audio/wav')

    logger.info(f'Http server: http://{args.host}:{args.port}/ttskit'.replace('0.0.0.0', 'localhost'))
    server = wsgi.WSGIServer((args.host, args.port), app)
    server.serve_forever()


if __name__ == '__main__':
    start_sever()

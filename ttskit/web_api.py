# author: kuangdd
# date: 2021/4/23
"""
### web_api
语音合成WEB接口。
构建简单的语音合成服务。

+ 简单使用
```python
from ttskit import web_api

web_api.app.run(host='0.0.0.0', port=2718, debug=False)
# 用POST或GET方法请求：http://localhost:2718/tts，传入参数text、audio、speaker。
# 例如GET方法请求：http://localhost:2718/tts?text=这是个例子&audio=2
```

+ 使用说明
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

import sys
import os

sys.path.append(os.path.abspath('..'))

from flask import Flask, request, jsonify, Response

import ttskit.sdk_api as sdk_api

app = Flask(__name__)


def parse_request(req_data):
    """解析请求数据并以json形式返回"""
    if req_data.method == 'POST':
        data = req_data.json
    elif req_data.method == 'GET':
        data = req_data.args
    else:  # POST
        data = req_data.get_json()
    return data


@app.route('/tts', methods=['POST', 'GET'])
def tts_web():
    """部署WEB的API接口。"""
    data = parse_request(request)
    text = data.get('text', '这是个样例')
    speaker = data.get('speaker', 'biaobei')
    audio = data.get('audio', '24')
    wav = sdk_api.tts_sdk(text=text, speaker=speaker, audio=audio)
    return Response(wav, mimetype='audio/wav')


if __name__ == "__main__":
    logger.info(__file__)
    app.run(host='0.0.0.0', port=2718, debug=False)

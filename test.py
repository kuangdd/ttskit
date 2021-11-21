#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/2/23
"""
test
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)


def test_sdk_api():
    from ttskit import sdk_api

    wav = sdk_api.tts_sdk('文本', audio='24')


def test_cli_api():
    from ttskit import cli_api

    args = cli_api.parse_args()
    cli_api.tts_cli(args)


def test_web_api():
    from ttskit import web_api

    web_api.app.run(host='0.0.0.0', port=2718, debug=False)
    # 用POST或GET方法请求：http://localhost:2718/tts，传入参数text、audio、speaker。
    # 例如GET方法请求：http://localhost:2718/tts?text=这是个例子&audio=2


def test_http_server():
    from ttskit import http_server

    http_server.start_sever()
    # 打开网页：http://localhost:9000/ttskit


if __name__ == "__main__":
    logger.info(__file__)
    # test_sdk_api()
    # test_cli_api()
    # test_web_api()
    test_http_server()

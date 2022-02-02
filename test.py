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
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '_'

    from ttskit import sdk_api
    text = '初始化。'
    wav = sdk_api.tts_sdk(text, speaker='biaobei', audio='24', processes=0)

    text = '欢迎使用语音合成工具箱这是二十个字的文本。' * 5
    wav = sdk_api.tts_sdk(text, speaker='biaobei', audio='24', processes=0)


def test_cli_api():
    from ttskit import cli_api

    args = cli_api.parse_args()
    cli_api.tts_cli(args)


def test_web_api():
    from ttskit import web_api

    web_api.app.run(host='0.0.0.0', port=2718, debug=False, processes=6, threaded=False)
    # 用POST或GET方法请求：http://localhost:2718/tts，传入参数text、audio、speaker。
    # 例如GET方法请求：http://localhost:2718/tts?text=这是个例子&audio=2


def test_http_server():
    from ttskit import http_server
    from multiprocessing import Process

    server = http_server.start_sever()
    # 单进程
    # server.serve_forever()
    # 多进程
    server.start()
    for i in range(6):
        p = Process(target=http_server.serve_forever, args=(server,))
        p.start()
    # 打开网页：http://localhost:9000/ttskit


if __name__ == "__main__":
    logger.info(__file__)
    from ttskit.sdk_api import load_models, _global_mspk_kwargs
    load_models(**_global_mspk_kwargs)
    # test_sdk_api()
    # test_cli_api()
    # test_web_api()
    # test_http_server()
    # import time
    # import threadpool
    # def sayhello(str):
    #     print("Hello ",str)
    #     time.sleep(2)
    #     return 'hello'
    #
    # name_list =['xiaozi','aa','bb','cc']
    # start_time = time.time()
    # pool = threadpool.ThreadPool(10)
    # requests = threadpool.makeRequests(sayhello, name_list)
    # out = [pool.putRequest(req) for req in requests]
    # print([req.args for req in requests])
    # print(out, requests)
    # pool.wait()
    # print('%d second'% (time.time()-start_time))

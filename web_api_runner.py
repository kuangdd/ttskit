#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2021-12-04
"""
"""
import argparse
import os


def set_args():
    """设置所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='_', type=str, help='设置预测时使用的显卡,使用CPU设置成_即可')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='IP地址')
    parser.add_argument('--port', type=int, default=7000, help='端口号')
    parser.add_argument('--processes', type=int, default=1, help='进程数')
    parser.add_argument('--threaded', type=int, default=1, help='是否开启线程')
    return parser.parse_args()


if __name__ == "__main__":
    print(__file__)
    from setproctitle import setproctitle
    from pathlib import Path

    setproctitle(Path(__file__).stem)

    args = set_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if args.device in {'_', '-1'} else args.device
    from ttskit.web_api import app

    app.run(host=args.host, port=args.port, debug=False, processes=args.processes, threaded=args.threaded)

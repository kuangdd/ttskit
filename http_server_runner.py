#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2021-12-04
"""
"""
from ttskit import http_server
from setproctitle import setproctitle
from pathlib import Path

if __name__ == "__main__":
    print(__file__)
    setproctitle(Path(__file__).stem)
    http_server.start_sever()

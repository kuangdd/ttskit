#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2021-12-05
"""
__init__的模块移动到core
"""
from . import sdk_api
from . import cli_api
from . import web_api
from . import http_server
from . import encoder
from . import mellotron
from . import waveglow
from . import melgan
from . import resource
from .sdk_api import tts_sdk as tts

if __name__ == "__main__":
    print(__file__)

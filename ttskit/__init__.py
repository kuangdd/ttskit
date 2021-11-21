"""
![ttskit](ttskit.png "ttskit")

## ttskit
Text To Speech Toolkit: 语音合成工具箱。

### 安装

```
pip install -U ttskit
```

- 注意
    * 可能需另外安装的依赖包：torch，版本要求torch>=1.6.0,<=1.7.1，根据自己的实际环境安装合适cuda或cpu版本的torch。
    * ttskit的默认音频采样率为22.5k。
    * 自行设置环境变量CUDA_VISIBLE_DEVICES以调用GPU，如果不设置，则默认调用0号GPU，没有GPU则使用CPU。
    * 默认用mspk模式的多发音人的语音合成模型，griffinlim声码器。

### 资源
使用ttskit的过程中会自动下载模型和语音资源。

如果下载太慢或无法下载，也可自行从百度网盘下载，把下载的资源合并到ttskit目录下（更新resource目录）。

链接：https://pan.baidu.com/s/13RPGNEKrCX3fgiGl7P5bpw

提取码：b7hw

### 快速使用
```
import ttskit

ttskit.tts('这是个示例', audio='14')

# 参数介绍
'''语音合成函数式SDK接口，函数参数全部为字符串格式。
text为待合成的文本。
speaker为发音人名称，可选名称为_reference_audio_dict；默认的发音人名称列表见resource/reference_audio/__init__.py。
audio为发音人参考音频，如果是数字，则调用内置的语音作为发音人参考音频；如果是语音路径，则调用audio路径的语音作为发音人参考音频。
注意：如果用speaker来选择发音人，请把audio设置为下划线【_】。
output为输出，如果以.wav结尾，则为保存语音文件的路径；如果以play开头，则合成语音后自动播放语音。
'''
```
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

__version__ = "0.3.2"

version_doc = """
### 版本
v{}
""".format(__version__)

readme_docs = [
    __doc__, version_doc,
    sdk_api.__doc__, cli_api.__doc__, web_api.__doc__, http_server.__doc__,
    resource.__doc__, encoder.__doc__, mellotron.__doc__, waveglow.__doc__, melgan.__doc__,
]


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

    ### 版本
    v0.3.6
    
### sdk_api
语音合成SDK接口。
本地函数式地调用语音合成。

+ 简单使用
```python
from ttskit import sdk_api

wav = sdk_api.tts_sdk('文本', audio='24')
```

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

+ 注意事项
    1. 模式mode
        - 可选：mspk、rtvc
        - 默认：mspk
    2. 声码器vocoder
        - 可选：melgan、griffinlim、waveglow
        - 默认：melgan
        - melgan控制参数
            * vocoder: melgan
        - griffinlim控制参数
            * vocoder: griffinlim
            * griffinlim_iters: 30
        - waveglow控制参数
            * vocoder: waveglow
            * sigma: 1.0
            * denoiser_strength: 1.2
    3. 参考音频audio
        - 可选：1-24的整数（内置的24个参考音频）、下划线_（）
        - POST请求接口：可传入wav音频的base64编码的字符串。
    4. 发音人speaker
        - 可选：Aibao、Aicheng、Aida、Aijia、Aijing、Aimei、Aina、Aiqi、Aitong、Aiwei、Aixia、Aiya、Aiyu、Aiyue、Siyue、Xiaobei、Xiaogang、Xiaomei、Xiaomeng、Xiaowei、Xiaoxue、Xiaoyun、Yina、biaobei

### resource
模型数据等资源。

audio
model
reference_audio

+ 内置发音人映射表

```python
_speaker_dict = {
    1: 'Aibao', 2: 'Aicheng', 3: 'Aida', 4: 'Aijia', 5: 'Aijing',
    6: 'Aimei', 7: 'Aina', 8: 'Aiqi', 9: 'Aitong', 10: 'Aiwei',
    11: 'Aixia', 12: 'Aiya', 13: 'Aiyu', 14: 'Aiyue', 15: 'Siyue',
    16: 'Xiaobei', 17: 'Xiaogang', 18: 'Xiaomei', 19: 'Xiaomeng', 20: 'Xiaowei',
    21: 'Xiaoxue', 22: 'Xiaoyun', 23: 'Yina', 24: 'biaobei'
}
```

### encoder
#### 声音编码器（encoder）
- 把语音音频编码为指定维度的向量。
- 向量的相似度反映音频音色的相似度。如果两个音频的编码向量相似度越高，则这两个音频的音色越接近。
- 编码向量主要用于控制发音的音色。

#### GE2E声音编码器
- 谷歌在上发布了GE2E算法的论文，详细介绍了其声纹识别技术的核心实现方法。
- 这是一种基于批（batch）的训练方法，这种基于批的训练，则是将同一批中每个说话者与其最相似的说话者的声纹特征变得不同。
- 论文通过理论和实验论证了，这种始终针对最困难案例进行优化的训练方式，能够极大地提升训练速度和效果。

### mellotron
#### 语音合成器（synthesizer）
- 把文本转为语音频谱数据。
- 语音合成器接收声音编码向量和文本数据，然后结合这些信息把文本转为语音频谱。
- 语音合成器的任务是把文本转为语音频谱，本质上是序列到序列的任务。
- 可以把文本看做一个一个字组成的序列，把语音频谱看做是由一个一个语音特征组成的序列，语音合成器就是把文字序列转为语音特征序列的桥梁。
- 语音合成器的关键就是怎样建立模型让文字准确的转为正确的读音，而且放在正确的位置，同时读音前后应当衔接自然，而且整个语音听起来也应当自然。
- 要实现这样的目标，应当做很有针对性的模型。

#### Mellotron语音合成器
- Mellotron是英伟达团队提出的语音合成模型，主要目标是做韵律风格转换和音乐生成。
- Mellotron可以更加精细化的调整韵律和音调，将基频信息引入模型刻画声调信息，基频是区别音高的主要元素。
- Mellotron模型的训练完全端到端化，不需要在数据集中含有音乐数据也可以生成音乐。
- Mellotron不需要对音调和文本进行人为的对齐就可以学到两者之间的对齐关系。
- Mellotron可以使用外部输入的注意力映射表，从而实现韵律迁移。

### waveglow
#### 声码器（vocoder）
- 把语音频谱数据转为语音信号。
- 语音信号和语音频谱数据并不是简单可以相互转换的数据，语音转为频谱是有信息丢失的，但是频谱记录了语音最主要的信息，可以通过其他技术手段把语音频谱尽可能逼真地逆变为语音信号，声码器就是这样的技术。
- 声码器是把声音特征转为语音信号的技术。
- 在语音合成任务中，声码器是负责把语音频谱转为语音信号。
- 通常语音频谱记录的语音信息并不是全面的，例如mel频谱只是记录了部分频段的幅度信息，缺失了相位信息，而且许多频率的信息也丢失了。
- 而声码器模型就是要从这样的频谱中尽可能准确全面地还原出语音信号。
- 现在通常的方案是用深度学习的方法来解决，针对语音特征和语音信号的关系进行建模。

#### Waveglow声码器
- WaveGlow是英伟达团队提出的一种依靠流的从梅尔频谱图合成高质量语音的模型。
- Waveglow贡献是基于流的网络，结合了Glow和WaveNet的想法，因此网络称为WaveGlow 。
- WaveGlow是一个生成模型，通过从分布采样中生成音频。
- WaveGlow易于实施，仅使用单个网络进行训练，仅使用似然损失函数进行训练。
- WaveGlow是兼顾生成速度快、生成质量高、稳定性强的模型。

#### MelGAN声码器
- MelGAN采用非自回归前馈卷积架构，在不引入额外蒸馏和感知损失的前提下，依然能够产生高质量的语音。
- MelGAN是一种利用生成对抗网络（Generative Adversarial Networks，GANs）合成时域波形的方法。
- MelGAN由于模型计算复杂度低，并行度高，因而合成速度极快。
- MelGAN是基于GAN的生成模型，因而主要由生成器和判别器组成。
- 生成器输入语音的压缩表示，如梅尔频谱，经过一层卷积后送入上采样层，上采样将梅尔频谱的序列长度匹配波形的频率。
- 对于判别器，主体是由卷积层和下采样层组成，并且采用多尺度架构，也就是不但对原始音频做判别，还将降频处理之后的音频馈送到判别器中进行判断。

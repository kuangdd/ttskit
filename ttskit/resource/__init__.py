# author: kuangdd
# date: 2021/4/25
"""
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
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

_speaker_dict = {
    1: 'Aibao', 2: 'Aicheng', 3: 'Aida', 4: 'Aijia', 5: 'Aijing',
    6: 'Aimei', 7: 'Aina', 8: 'Aiqi', 9: 'Aitong', 10: 'Aiwei',
    11: 'Aixia', 12: 'Aiya', 13: 'Aiyu', 14: 'Aiyue', 15: 'Siyue',
    16: 'Xiaobei', 17: 'Xiaogang', 18: 'Xiaomei', 19: 'Xiaomeng', 20: 'Xiaowei',
    21: 'Xiaoxue', 22: 'Xiaoyun', 23: 'Yina', 24: 'biaobei'
}

if __name__ == "__main__":
    logger.info(__file__)
    for line in sorted(Path(__file__).parent.glob('*')):
        print(line.name)

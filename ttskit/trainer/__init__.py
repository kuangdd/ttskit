# author: kuangdd
# date: 2021/4/25
"""
### trainer
训练模型的模块。

#### 使用方法
```
from ttskit.trainer import mellotron_train as mttr
mttr.train()
```
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

if __name__ == "__main__":
    logger.info(__file__)
    for line in sorted(Path(__file__).parent.glob('*')):
        print(line.name)

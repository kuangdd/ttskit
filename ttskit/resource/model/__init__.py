# author: kuangdd
# date: 2021/4/25
"""
### model
ge2e.kuangdd.pt
mellotron.kuangdd-rtvc.pt
mellotron.kuangdd-rtvc.hparams.json
mellotron.kuangdd-mspk.pt
mellotron.kuangdd-mspk.hparams.json
waveglow.kuangdd.pt
melgan.kuangdd.pt
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

if __name__ == "__main__":
    logger.info(__file__)
    for line in sorted(Path(__file__).parent.glob('*')):
        print(line.name)

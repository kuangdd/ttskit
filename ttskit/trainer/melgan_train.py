import os
import time
import logging
import argparse
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ttskit.melgan.utils.train import train
from ttskit.melgan.utils.hparams import HParam
from ttskit.melgan.utils.writer import MyWriter
from ttskit.melgan.datasets.dataloader import create_dataloader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'melgan_config.yaml'),
                        help="yaml file for configuration")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file to resume training")
    parser.add_argument('-n', '--name', type=str, default='workplace/melgan-samples',
                        help="name of the model for logging, saving checkpoint")
    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, 'r', encoding='utf8') as f:
        hp_str = ''.join(f.readlines())

    pt_dir = os.path.join(args.name, hp.log.chkpt_dir)
    log_dir = os.path.join(args.name, hp.log.log_dir)
    os.makedirs(pt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, '%s-%d.log' % (os.path.basename(args.name), time.time()))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    writer = MyWriter(hp, log_dir)

    assert hp.audio.hop_length == 256, \
        'hp.audio.hop_length must be equal to 256, got %d' % hp.audio.hop_length
    assert hp.data.train != '' and hp.data.validation != '', \
        'hp.data.train and hp.data.validation can\'t be empty: please fix %s' % args.config

    trainloader = create_dataloader(hp, args, True)
    valloader = create_dataloader(hp, args, False)

    train(args, pt_dir, args.checkpoint_path, trainloader, valloader, writer, logger, hp, hp_str)

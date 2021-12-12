import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', type=str, default=r'workplace/samples/metadata.csv',
                        help='directory to save checkpoints')
    parser.add_argument('-o', '--output_directory', type=str, default=r"workplace/mellotron-mspk-samples",
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str, default='tensorboard',
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams_json', type=str,
                        default='{"batch_size":4,"iters_per_checkpoint":100,"learning_rate":0.0001,"dataloader_num_workers":0}',
                        required=False, help='comma separated name=value pairs')
    parser.add_argument('--hparams_level', type=int, default=2,
                        required=False, help='hparams scale')
    parser.add_argument("--cuda", type=str, default='0,1,2,3,4,5,6,7,8,9',
                        help='设置CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()

    return args


args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

import yaml
import torch

from ttskit.mellotron.hparams import create_hparams
from ttskit.mellotron.train import train, json_dump, yaml_dump

if __name__ == '__main__':
    try:
        from setproctitle import setproctitle

        setproctitle('ttskit-mellotron-train')
    except ImportError:
        pass

    hparams = create_hparams(args.hparams_json, level=args.hparams_level)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    meta_folder = os.path.join(args.output_directory, 'metadata')
    os.makedirs(meta_folder, exist_ok=True)

    stem_path = os.path.join(meta_folder, "args")
    obj = args.__dict__
    json_dump(obj, f'{stem_path}.json')
    yaml_dump(obj, f'{stem_path}.yml')

    print('{}\nargs:'.format('-' * 50))
    print(yaml.dump(args.__dict__))

    print('{}\nhparams:'.format('-' * 50))
    print(yaml.dump({k: v for k, v in hparams.items()}))

    train(hparams=hparams, **args.__dict__)

import os
import time
import argparse
import math
from pathlib import Path
import json
import shutil
import numpy as np
import multiprocessing as mp
import argparse
import yaml
import random

from numpy import finfo
from tqdm import tqdm
import aukit
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from .distributed import apply_gradient_allreduce
from .model import load_model
from .data_utils import TextMelLoader, TextMelCollate
from .loss_function import Tacotron2Loss
from .logger import Tacotron2Logger
from .hparams import create_hparams
from .utils import inv_linearspectrogram
from .plotting_utils import plot_mel_alignment_gate_audio
from .audio_processing import griffin_lim, dynamic_range_decompression
from .text.symbols import symbols

_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def json_dump(obj, path):
    obj = {k: v for k, v in obj.items()}
    if os.path.isfile(path):
        dt = json.load(open(path, encoding="utf8"))
        if obj != dt:
            path = "{}_{}.json".format(os.path.splitext(path)[0], time.strftime("%Y%m%d-%H%M%S"))
    json.dump(obj, open(path, "wt", encoding="utf8"), indent=4, ensure_ascii=False)


def yaml_dump(obj, path):
    with open(path, "wt", encoding='utf8') as fout:
        yaml.dump(obj, fout, default_flow_style=False, encoding='utf-8', allow_unicode=True)


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(input_directory, hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(os.path.join(input_directory, 'train.txt'), hparams, mode=hparams.train_mode)
    valset = TextMelLoader(os.path.join(input_directory, 'validation.txt'), hparams,
                           speaker_ids=trainset.speaker_ids, mode=hparams.train_mode)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=hparams.dataloader_num_workers, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn, train_sampler


def prepare_directories_and_logger(output_directory, log_directory, rank, hparams=None):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory), hparams=hparams)
    else:
        logger = None
    return logger


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}".format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank, outdir=Path(), hparams=None):
    """Handles all the validation scoring and printing"""
    save_flag = True
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        # fixme num_workers>=1则EOFError: Ran out of input，mp.spawn的报错。
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=0,
                                shuffle=True, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)  # shuffle=False,

        val_loss = 0.0
        for i, batch in enumerate(tqdm(val_loader, 'validate', ncols=100)):
            x, y = model.parse_batch(batch)  # y: 2部分
            # x: (text_padded, input_lengths, mel_padded, max_len, output_lengths, speaker_ids, f0_padded),
            # y: (mel_padded, gate_padded)
            # x:
            # torch.Size([4, 64])
            # torch.Size([4])
            # torch.Size([4, 401, 347])

            # y:
            # torch.Size([4, 401, 439])
            # torch.Size([4, 439])

            y_pred = model(x)  # y_pred: 4部分
            # y_pred:
            # torch.Size([4, 401, 439])
            # torch.Size([4, 401, 439])
            # torch.Size([4, 439])
            # torch.Size([4, 439, 114])

            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = y_pred
            loss = criterion(y_pred, y)
            if outdir and save_flag:
                curdir = outdir.joinpath('validation', f'{iteration:06d}-{loss.data.cpu().numpy():.4f}')
                curdir.mkdir(exist_ok=True, parents=True)
                idx = random.randint(0, alignments.size(0) - 1)

                gate_output = gate_outputs[idx].data.cpu().numpy()
                end_idx = np.argmax(gate_output > 0.5) or gate_output.shape[0]
                mel = mel_outputs_postnet[idx][:, :end_idx].unsqueeze(0)
                wav_outputs = valset.stft.griffin_lim(mel)
                wav_output = wav_outputs[0]  # .cpu().numpy()
                aukit.save_wav(wav_output, curdir.joinpath('griffinlim_pred.wav'), sr=hparams.sampling_rate)

                mel_targets = y[0]
                gate_targets = y[1]
                gate_target = gate_targets[idx].data.cpu().numpy()
                end_idx = np.argmax(gate_target > 0.5) or gate_target.shape[0]
                mel = mel_targets[idx][:, :end_idx].unsqueeze(0)
                wav_inputs = valset.stft.griffin_lim(mel)
                wav_input = wav_inputs[0]  # .cpu().numpy()
                aukit.save_wav(wav_input, curdir.joinpath('griffinlim_true.wav'), sr=hparams.sampling_rate)

                plot_mel_alignment_gate_audio(target=mel_targets[idx].cpu().numpy(),
                                              mel=mel_outputs[idx].cpu().numpy(),
                                              alignment=alignments[idx].cpu().numpy().T,
                                              gate=torch.sigmoid(gate_outputs[idx]).cpu().numpy(),
                                              audio=wav_output[::hparams.sampling_rate // 100])
                plt.savefig(curdir.joinpath('figure.png'))
                plt.close()

                save_flag = False

            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, reduced_val_loss))
        logger.log_validation(val_loss, model, y, y_pred, iteration, x)


def train(input_directory, output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams, **kwargs):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    # torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = Tacotron2Loss()

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank, hparams=hparams)

    # 记录训练的元数据。
    meta_folder = os.path.join(output_directory, 'metadata')
    os.makedirs(meta_folder, exist_ok=True)

    trpath = os.path.join(meta_folder, "train.txt")
    vapath = os.path.join(meta_folder, "validation.txt")
    with open(trpath, 'wt', encoding='utf8') as fout_tr, open(vapath, 'wt', encoding='utf8') as fout_va:
        lines = open(input_directory, encoding='utf8').readlines()
        val_ids = set(np.random.choice(list(range(len(lines))), hparams.batch_size * 2, replace=False))
        for num, line in enumerate(lines):
            parts = line.strip().split('\t')
            abspath = os.path.join(os.path.dirname(os.path.abspath(input_directory)), parts[0]).replace('\\', '/')
            text = parts[1]
            if len(parts) >= 3:
                speaker = parts[2]
            else:
                speaker = '0'
            out = f'{abspath}\t{text}\t{speaker}\n'
            if num in val_ids:
                fout_va.write(out)
            else:
                fout_tr.write(out)

    train_loader, valset, collate_fn, train_sampler = prepare_dataloaders(meta_folder, hparams)

    stem_path = os.path.join(meta_folder, "locals")
    obj = {k: (str(v) if isinstance(v, Path) else v)
           for k, v in locals().items() if isinstance(v, (int, float, str, Path, bool))}
    json_dump(obj, f'{stem_path}.json')
    yaml_dump(obj, f'{stem_path}.yml')

    stem_path = os.path.join(meta_folder, "speakers")
    obj = {k: v for k, v in valset.speaker_ids.items()}
    json_dump(obj, f'{stem_path}.json')
    yaml_dump(obj, f'{stem_path}.yml')

    stem_path = os.path.join(meta_folder, "hparams")
    obj = {k: v for k, v in hparams.items()}
    json_dump(obj, f'{stem_path}.json')
    yaml_dump(obj, f'{stem_path}.yml')

    stem_path = os.path.join(meta_folder, "symbols")
    obj = {w: i for i, w in enumerate(symbols)}
    json_dump(obj, f'{stem_path}.json')
    yaml_dump(obj, f'{stem_path}.yml')

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    iteration_start = iteration

    checkpoint_folder = os.path.join(output_directory, 'checkpoint')
    os.makedirs(checkpoint_folder, exist_ok=True)

    model.train()
    is_overflow = False
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch-{epoch}", ncols=100)):
            start = time.perf_counter()
            if iteration > 0 and iteration % hparams.learning_rate_anneal == 0:
                learning_rate = max(
                    hparams.learning_rate_min, learning_rate * 0.5)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(x)

            loss = criterion(y_pred, y)
            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()

            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), hparams.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()
            duration = time.perf_counter() - start
            if not is_overflow and rank == 0:
                logger.log_training(
                    reduced_loss, grad_norm, learning_rate, duration, iteration)

            if not is_overflow and ((iteration % hparams.iters_per_checkpoint == 0) or (iteration == iteration_start)):
                print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, reduced_loss, grad_norm, duration))
                validate(model, criterion, valset, iteration,
                         hparams.batch_size, n_gpus, collate_fn, logger,
                         hparams.distributed_run, rank, outdir=Path(output_directory), hparams=hparams)
                if rank == 0:
                    checkpoint_path = os.path.join(checkpoint_folder, "mellotron-{:06d}.pt".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path)

            iteration += 1


if __name__ == '__main__':
    print(__file__)

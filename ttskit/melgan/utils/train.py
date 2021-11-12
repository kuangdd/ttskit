import os
import math
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import traceback

from model.generator import Generator
from model.multiscale import MultiScaleDiscriminator
from .utils import get_commit_hash
from .validation import validate


def train(args, pt_dir, chkpt_path, trainloader, valloader, writer, logger, hp, hp_str):
    model_g = Generator(hp.audio.n_mel_channels).cuda()
    model_d = MultiScaleDiscriminator().cuda()

    optim_g = torch.optim.Adam(model_g.parameters(),
        lr=hp.train.adam.lr, betas=(hp.train.adam.beta1, hp.train.adam.beta2))
    optim_d = torch.optim.Adam(model_d.parameters(),
        lr=hp.train.adam.lr, betas=(hp.train.adam.beta1, hp.train.adam.beta2))

    githash = get_commit_hash()

    init_epoch = -1
    step = 0

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model_g.load_state_dict(checkpoint['model_g'])
        model_d.load_state_dict(checkpoint['model_d'])
        optim_g.load_state_dict(checkpoint['optim_g'])
        optim_d.load_state_dict(checkpoint['optim_d'])
        step = checkpoint['step']
        init_epoch = checkpoint['epoch']

        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint. Will use new.")

        if githash != checkpoint['githash']:
            logger.warning("Code might be different: git hash is different.")
            logger.warning("%s -> %s" % (checkpoint['githash'], githash))

    else:
        logger.info("Starting new training run.")

    # this accelerates training when the size of minibatch is always consistent.
    # if not consistent, it'll horribly slow down.
    torch.backends.cudnn.benchmark = True

    try:
        model_g.train()
        model_d.train()
        for epoch in itertools.count(init_epoch+1):
            if epoch % hp.log.validation_interval == 0:
                with torch.no_grad():
                    validate(hp, args, model_g, model_d, valloader, writer, step)

            trainloader.dataset.shuffle_mapping()
            loader = tqdm.tqdm(trainloader, desc='Loading train data')
            for (melG, audioG), (melD, audioD) in loader:
                melG = melG.cuda()
                audioG = audioG.cuda()
                melD = melD.cuda()
                audioD = audioD.cuda()

                # generator
                optim_g.zero_grad()
                fake_audio = model_g(melG)[:, :, :hp.audio.segment_length]
                disc_fake = model_d(fake_audio)
                disc_real = model_d(audioG)
                loss_g = 0.0
                for (feats_fake, score_fake), (feats_real, _) in zip(disc_fake, disc_real):
                    loss_g += torch.mean(torch.sum(torch.pow(score_fake - 1.0, 2), dim=[1, 2]))
                    for feat_f, feat_r in zip(feats_fake, feats_real):
                        loss_g += hp.model.feat_match * torch.mean(torch.abs(feat_f - feat_r))

                loss_g.backward()
                optim_g.step()

                # discriminator
                fake_audio = model_g(melD)[:, :, :hp.audio.segment_length]
                fake_audio = fake_audio.detach()
                loss_d_sum = 0.0
                for _ in range(hp.train.rep_discriminator):
                    optim_d.zero_grad()
                    disc_fake = model_d(fake_audio)
                    disc_real = model_d(audioD)
                    loss_d = 0.0
                    for (_, score_fake), (_, score_real) in zip(disc_fake, disc_real):
                        loss_d += torch.mean(torch.sum(torch.pow(score_real - 1.0, 2), dim=[1, 2]))
                        loss_d += torch.mean(torch.sum(torch.pow(score_fake, 2), dim=[1, 2]))

                    loss_d.backward()
                    optim_d.step()
                    loss_d_sum += loss_d

                step += 1
                # logging
                loss_g = loss_g.item()
                loss_d_avg = loss_d_sum / hp.train.rep_discriminator
                loss_d_avg = loss_d_avg.item()
                if any([loss_g > 1e8, math.isnan(loss_g), loss_d_avg > 1e8, math.isnan(loss_d_avg)]):
                    logger.error("loss_g %.01f loss_d_avg %.01f at step %d!" % (loss_g, loss_d_avg, step))
                    raise Exception("Loss exploded")

                if step % hp.log.summary_interval == 0:
                    writer.log_training(loss_g, loss_d_avg, step)
                    loader.set_description("g %.04f d %.04f | step %d" % (loss_g, loss_d_avg, step))

            if epoch % hp.log.save_interval == 0:
                save_path = os.path.join(pt_dir, '%s_%s_%04d.pt'
                    % (args.name, githash, epoch))
                torch.save({
                    'model_g': model_g.state_dict(),
                    'model_d': model_d.state_dict(),
                    'optim_g': optim_g.state_dict(),
                    'optim_d': optim_d.state_dict(),
                    'step': step,
                    'epoch': epoch,
                    'hp_str': hp_str,
                    'githash': githash,
                }, save_path)
                logger.info("Saved checkpoint to: %s" % save_path)

    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
